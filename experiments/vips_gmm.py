import numpy as np
from scipy.stats import multivariate_normal as normal_pdf
from scipy import log, exp
import torch as th
import matplotlib.pyplot as plt
from rat_spn import RatSpn, RatSpnConfig
from distributions import RatNormal
from experiments.train_mnist import count_params
from experiments.mnist_gen_train import time_delta
from torch import optim
import torch.distributions as dist
import gif
import os
import time
from utils import *


def build_ratspn(F, I, bounds):
    config = RatSpnConfig()
    config.C = 1
    config.F = F
    config.R = 3
    config.D = int(np.log2(F))
    config.I = 4
    config.S = 3
    config.dropout = 0.0
    config.leaf_base_class = RatNormal
    config.leaf_base_kwargs = {'min_mean': float(bounds[0]), 'max_mean': float(bounds[1])}
    model = RatSpn(config)
    count_params(model)
    return model


def target_callback(x):
    return target_mixture.evaluate(x, dims=None, return_log=True, has_rep_dim=True)  # + 200.0


def scale_to_grid(tensor):
    return (tensor - min_x) / (max_x - min_x) * grid_points


def scale_to_x(tensor):
    return (tensor / grid_points) * (max_x - min_x) + min_x


def exp_view(tensor: th.Tensor):
    return tensor.exp().view(grid_points, grid_points).T.cpu()


def dist_imshow(handle, probs, apply_exp_view=True, **kwargs):
    # handle is either an axis or plt
    if apply_exp_view:
        probs = exp_view(probs)
    handle.imshow(forplot(probs), extent=[min_x, max_x, max_x, min_x], **kwargs)


def forplot(tensor: th.Tensor, scale=False):
    tensor = tensor.detach().cpu().numpy()
    if scale:
        tensor = scale_to_grid(tensor)
    return tensor


@gif.frame
def gif_frame(probs):
    fig, (ax1) = plt.subplots(1, figsize=(10, 10), dpi=200)
    ax1.imshow(forplot(exp_view(probs)))
    ax1.set_title(f"RatSpn distribution at step {step}")


@gif.frame
def gif_target_dist(model_probs, step, leaf_mpe, root_children_mpe):
    plot_target_dist(model_probs, True, step, leaf_mpe, root_children_mpe)


def plot_target_dist(model_probs=None, noshow=False, step=None, leaf_mpe=None, root_children_mpe=None):
    target_probs = exp_view(target_mixture.evaluate(grid, dims=None, return_log=True))
    if model_probs is not None:
        model_probs = exp_view(model_probs)
    if isinstance(target_mixture, IndGmm):
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 15), dpi=100)
        if model_probs is None:
            dist_imshow(ax1, target_probs, apply_exp_view=False)
        else:
            dist_imshow(ax1, target_probs, apply_exp_view=False, alpha=0.5, cmap='cividis')
            dist_imshow(ax1, model_probs, apply_exp_view=False, alpha=0.7)
        if root_children_mpe is not None:
            root_children_mpe = th.einsum('...ij -> ...ji', root_children_mpe).flatten(0, -2)
            root_children_mpe = forplot(root_children_mpe)
            ax1.scatter(root_children_mpe[:, 0], root_children_mpe[:, 1], s=1, color='r', label='Modes')
        ax1.set_title(
            f"Target distribution with "
            f"{', '.join([f'{target_mixture.num_components[i]} components over {dim}' for i, dim in enumerate(['x', 'y'])])}."
        )
        ax2.plot(scale_to_x(np.arange(grid_points)), forplot(target_probs.sum(0)), color='b', label='Target dist')
        ax2.set_ylim(None, 0.25)
        ax3.plot(scale_to_x(np.arange(grid_points)), forplot(target_probs.sum(1)), color='b', label='Target dist')
        ax3.set_ylim(None, 0.25)
        if model_probs is not None:
            ax2.plot(scale_to_x(np.arange(grid_points)), forplot(model_probs.sum(0)), color='r',
                     label='Model dist')
            ax3.plot(scale_to_x(np.arange(grid_points)), forplot(model_probs.sum(1)), color='r',
                     label='Model dist')
        if leaf_mpe is not None:
            leaf_mpe = th.einsum('...ij -> ...ji', leaf_mpe).flatten(0, -2)
            leaf_mpe = forplot(leaf_mpe)
            ax2.vlines(leaf_mpe[:, 0], ymin=0, ymax=0.1, linestyles='-', alpha=0.7, label='Modes', colors='r')
            ax3.vlines(leaf_mpe[:, 1], ymin=0, ymax=0.1, linestyles='-', alpha=0.7, label='Modes', colors='r')

        ax2.set_title("Marginal target distribution over x")
        ax3.set_title("Marginal target distribution over y")
        ax2.legend()
        ax3.legend()
        if step is not None:
            fig.suptitle(f"VIPS at step {step}")
    else:
        plt.imshow(target_probs)
        plt.title(f"Target distribution with {num_true_components} components.")
    if not noshow:
        plt.show()


class Target:
    """
        Target distribution to fit. Is kind of like a prison.
    """
    def __init__(self, num_dims):
        self.num_dims = num_dims
        self.wall_thickness = 2
        self.num_cells_per_dim = 8
        self.target_support = np.asarray([[-50, 50], [-50, 50]])  # x and y support
        self.l_bound = self.target_support[:, 0]
        self.u_bound = self.target_support[:, 1]
        self.cell_size = (self.u_bound - self.l_bound) / self.num_cells_per_dim

    def evaluate(self, point):
        dist_ahead = point % self.cell_size
        dist_behind = self.cell_size - (point % self.cell_size)
        min_dist = np.min([dist_behind, dist_ahead], axis=0)
        is_in_wall = np.max(min_dist < self.wall_thickness / 2, axis = 1)
        return is_in_wall


class IndGmm:
    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions
        self.num_components = [0] * num_dimensions
        self.mixture = []
        self.weights = []

    def add_dim(self, means, stds):
        assert means.dim() == 1
        d = dist.Normal(means, stds, validate_args=False)
        self.mixture.append(d)
        self.weights.append(th.log(th.ones(len(means)) / len(means)))
        self.num_components[dim] = len(means)

    def evaluate(self, x: th.Tensor, dims: list = None, return_log=False, has_rep_dim=False):
        """

        Args:
            x: Shape [ic, s, self.config.R, n, w, self.config.F]
                Samples of the SPN to evaluate the target log-probs of
            dims:
            return_log:

        Returns:

        """
        if dims is None:
            dims = [i for i in range(self.num_dimensions)]
        x = x.cpu()

        if False and probs is not None:
            for key, value in {'SPN': probs, 'target': target_probs}.items():
                fig, (ax1) = plt.subplots(1, figsize=(10, 10), dpi=100)
                ax1.imshow(forplot(exp_view(value)))
                flat = x.flatten(0, -2)
                nan_mask = flat.isnan()
                no_nans = flat[~nan_mask.sum(1).bool(), :]
                solo_x = flat[:, 0][nan_mask[:, 1]]
                solo_y = flat[:, 1][nan_mask[:, 0]]
                if no_nans.shape[0] > 0:
                    ax1.scatter(forplot(flat[:, 0], True), forplot(flat[:, 1], True), s=1, color='r')
                if solo_x.shape[0] > 0:
                    ax1.vlines(forplot(solo_x, True), ymin=0, ymax=grid_points - 1, color='r', alpha=0.3)
                if solo_y.shape[0] > 0:
                    ax1.hlines(forplot(solo_y, True), xmin=0, xmax=grid_points - 1, color='r', alpha=0.3)
                ax1.set_title(f"Samples of {'root node children (product nodes)' if x.shape[1] == 1 else 'leaf nodes'} "
                              f"in {key} distribution.")
                plt.show()

        log_probs = []
        for i, dim in enumerate(dims):
            component_probs = self.mixture[dim].log_prob(x[..., [i]])
            mixture_prob = th.logsumexp(component_probs + self.weights[dim], dim=-1)
            log_probs.append(mixture_prob)

        if has_rep_dim:
            log_probs = th.stack(log_probs, dim=-1)
        else:
            log_probs = th.sum(th.stack(log_probs), dim=0)
        return log_probs if return_log else log_probs.exp()


class GMM:
    # Class Constructor
    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions
        self.num_components = 0
        self.weights = np.array([])
        self.mixture = []

    def add_component(self, mean, cov):
        self.mixture.append(normal_pdf(mean, cov))
        self.weights = np.ones(len(self.mixture)) / len(self.mixture)
        self.num_components+=1

    def evaluate(self, x, return_log=False):
        if return_log:
            if x.ndim == 1:
                logpdf = np.empty((self.num_components))
            else:
                logpdf = np.empty((self.num_components, x.shape[0]))
            for i in range(0, self.num_components):
                logpdf[i] = self.mixture[i].logpdf(x) + np.log(self.weights[i])
            maxLogPdf = np.max(logpdf,0)
            if x.ndim == 1:
                return log(sum(exp(logpdf - maxLogPdf))) + maxLogPdf
            else:
                return log(sum(exp(logpdf - maxLogPdf[np.newaxis,:]))) + maxLogPdf
        else:
            if x.ndim == 1:
                pdf = np.empty((self.num_components))
            else:
                pdf = np.empty((self.num_components, x.shape[0]))
            for i in range(0, self.num_components):
                pdf[i] = self.mixture[i].pdf(x) * self.weights[i]
            return sum(pdf, 0)

    def sample(self, n):
        sampled_components = np.random.choice(a=self.num_components, size=n, replace=True, p = self.weights)
        counts = np.bincount(sampled_components)
        different_components = np.where(counts > 0)[0]
        # sample from each chosen component
        samples = np.vstack(
            [self.mixture[idx].rvs(counts[idx]).reshape(-1, self.num_dimensions) for idx in different_components])

        # shuffle the samples in order to make sure that we are unbiased when using just the last N samples
        indices = np.arange(len(sampled_components))
        np.random.shuffle(indices)
        return samples[indices]

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def get_numpy_means(self):
        return np.asarray([comp.mean for comp in self.mixture])

    def get_numpy_covs(self):
        return np.asarray([comp.cov for comp in self.mixture])


def build_GMM_lnpdf(num_dimensions, num_true_components, prior_variance=1e3):
    prior = normal_pdf(np.zeros(num_dimensions), prior_variance * np.eye(num_dimensions))
    prior_chol = np.sqrt(prior_variance) * np.eye(num_dimensions)
    target_mixture = GMM(num_dimensions)
    for i in range(0, num_true_components):
        this_cov = 0.1 * np.random.normal(0, num_dimensions, (num_dimensions * num_dimensions)).reshape(
            (num_dimensions, num_dimensions))
        this_cov = this_cov.transpose().dot(this_cov)
        this_cov += 1 * np.eye(num_dimensions)
        this_mean = 100 * (np.random.random(num_dimensions) - 0.5)
        target_mixture.add_component(this_mean, this_cov)

    target_mixture.set_weights(np.ones(num_true_components) / num_true_components)
    def target_lnpdf(theta, without_prior=False):
        theta = np.atleast_2d(theta)
        target_lnpdf.counter += len(theta)
        if without_prior:
            return np.squeeze(target_mixture.evaluate(theta, return_log=True) - prior.logpdf(theta))
        else:
            return np.squeeze(target_mixture.evaluate(theta, return_log=True))

    target_lnpdf.counter = 0
    return [target_lnpdf, prior, prior_chol, target_mixture]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, default='vips_gmm_test',
                        help='Experiment name. The results dir will contain it.')
    parser.add_argument('--seed', '-s', type=int, nargs='+', required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='cuda or cpu')
    parser.add_argument('--ent_approx_sample_size', '-samples', type=int, default=5)
    parser.add_argument('--results_dir', type=str, default='../../spn_experiments',
                        help='The base directory where the directory containing the results will be saved to.')
    parser.add_argument('--no_resp_grad', action='store_true',
                        help="If True, approximation of responsibilities is done with grad disabled.")
    parser.add_argument('--entropy', '-ent', action='store_true',
                        help="If True, create a new SPN and train it to increase entropy only.")
    parser.add_argument('--vips', action='store_true',
                        help="If True, fit model to target dist using our flavor of VIPS.")
    parser.add_argument('--gif_duration', '-gif_d', type=int, default=30)
    parser.add_argument('--make_gif', '-gif', action='store_true',
                        help="Create gif of plots")
    parser.add_argument('--init_weight_kl_bound', '-w_kl', type=float, default=0.0,
                        help='Initial KL bound on all sum weights.')
    parser.add_argument('--weight_update_start', '-w_start', type=int, default=0, help='Start weight updates at step.')
    parser.add_argument('--init_leaf_kl_bound', '-l_kl', type=float, default=1e-2,
                        help='Initial KL bound on all sum weights.')
    args = parser.parse_args()

    assert not (args.entropy and args.vips)

    for d in [args.results_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    probs = None
    num_dimensions = 2
    num_true_components = 10
    if False:
        target_gmm_prior_variance = 1e3
        [target_lnpdf, _, _, target_mixture] = build_GMM_lnpdf(num_dimensions, num_true_components,
                                                               target_gmm_prior_variance)
    else:
        target_mixture = IndGmm(num_dimensions)
        if True:
            dist_params = {
                0: {
                    'mean': th.as_tensor([-40, -20, -10, 10, 30, 40]),
                    'std': th.as_tensor([2, 4, 2, 3, 2, 3]),
                },
                1: {
                    'mean': th.as_tensor([-35, -20, 0, 15, 35]),
                    'std': th.as_tensor([3, 2, 3, 4, 2])
                }
            }
        elif False:
            dist_params = {
                0: {
                    'mean': th.as_tensor([-40, -20, -10, 10, 30, 40]),
                    'std': th.as_tensor([5] * 6),
                },
                1: {
                    'mean': th.as_tensor([-35, -15, 15, 35]),
                    'std': th.as_tensor([5] * 4)
                }
            }
        else:
            dist_params = {
                0: {
                    'mean': th.as_tensor([-30]),
                    'std': th.as_tensor([1] * 1),
                },
                1: {
                    'mean': th.as_tensor([35]),
                    'std': th.as_tensor([3] * 1)
                }
            }
        for dim, params in dist_params.items():
            target_mixture.add_dim(params['mean'], params['std'])

    grid_points = 501
    min_x = -50
    max_x = -min_x
    x = th.linspace(min_x, max_x, grid_points)
    grid = th.stack(th.meshgrid((x, x), indexing='ij'), dim=-1)
    grid = grid.reshape(-1, 2)

    if args.vips:
        plot_target_dist()

    for seed in args.seed:
        th.manual_seed(seed)

        if False and args.vips:
            load_path = os.path.join(args.results_dir, 'high_ent_ratspn.pt')
        else:
            load_path = None
        if load_path is None:
            model = build_ratspn(
                F=2,
                I=int(num_true_components * 1.0),
                bounds=(min_x, max_x),
            ).to(args.device)
        else:
            model = th.load(load_path, map_location=args.device)
            # Only for high_ent_ratspn.pt
            w = model.layer_index_to_obj(model.max_layer_index).weight_param.data
            w = w.unsqueeze(0)
            model.layer_index_to_obj(model.max_layer_index).weight_param = th.nn.Parameter(w)

        grid_tensor = th.as_tensor(grid, device=model.device, dtype=th.float)

        if True:
            with th.no_grad():
                model_probs = model(grid_tensor)
                if args.vips:
                    plot_target_dist(model_probs)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        if args.make_gif:
            if args.entropy:
                fps = 5
                gif_duration = args.gif_duration  # seconds
                n_steps = args.steps
                n_frames = fps * gif_duration
                make_frame_every = int(n_steps / n_frames)
                save_path = os.path.join(
                    args.results_dir,
                    f"{'no_grad' if args.no_resp_grad else 'with_grad'}"
                    f"_samples{args.ent_approx_sample_size}"
                    f"_seed{seed}.gif"
                )
            else:
                fps = 10
                # n_steps = 151
                # make_frame_every = 1
                gif_duration = args.gif_duration  # seconds
                n_steps = args.steps
                n_frames = fps * gif_duration
                make_frame_every = int(n_steps / n_frames)

                save_path = os.path.join(
                    args.results_dir, f"{args.exp_name}.gif"
                )
        else:
            n_steps = args.steps
            make_frame_every = 10 # 5000
        print(f"Running for {n_steps} steps, making a frame every {make_frame_every} steps.")

        def bookmark():
            pass
        frames = []
        losses = []
        t_start = time.time()
        plot_at = 150

        def verbose_callback(step):
            return False
            # return step == plot_at
            # return True

        def step_callback(step):
            global t_start, losses, log
            if step % make_frame_every == 0:
                with th.no_grad():
                    probs = model(grid_tensor)
                if args.make_gif:
                    if args.vips:
                        frame = gif_target_dist(
                            model_probs=probs, step=step,
                            leaf_mpe=model.mpe(layer_index=0),
                            root_children_mpe=model.mpe(layer_index=model.max_layer_index-1),
                        )
                    else:
                        frame = gif_frame(probs)
                    frames.append(frame)
                elif True:
                    if args.vips:
                        plot_target_dist(
                            model_probs=probs, noshow=False, step=step,
                            leaf_mpe=model.mpe(layer_index=0),
                            root_children_mpe=model.mpe(layer_index=model.max_layer_index - 1),
                        )
                    else:
                        plt.imshow(forplot(exp_view(probs)), extent=[min_x, max_x, max_x, min_x])
                        plt.show()
                t_delta = np.around(time.time() - t_start, 2)
                if step > 0:
                    print(f"Time delta: {time_delta(t_delta)} - Step {step}"
                          f"{f' - Avg. loss: {round(np.mean(losses), 2)}' if len(losses) > 0 else ''}")
                losses = []
                t_start = time.time()

        if args.vips:
            log_dict = model.vips(
                target_dist_callback=target_callback,
                steps=n_steps,
                step_callback=step_callback,
                sample_size=50,
                init_weight_kl_bound=args.init_weight_kl_bound,
                init_leaf_kl_bound=args.init_leaf_kl_bound,
                weight_update_start=args.weight_update_start,
                verbose=verbose_callback,
            )
        elif args.entropy:
            # th.set_anomaly_enabled(True)
            for step in range(int(n_steps)):
                ent, log = model.vi_entropy_approx(
                    sample_size=args.ent_approx_sample_size,
                    grad_thru_resp=not args.no_resp_grad,
                    verbose=True,
                )
                loss = -ent.mean()
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step_callback(step)
        else:
            print("No experiment specified")
            exit()

        if args.make_gif:
            gif.save(frames, save_path, duration=1/fps, unit='s')

        if args.vips:
            plot_target_dist(
                model_probs=probs, noshow=False, step=n_steps-1,
                leaf_mpe=model.mpe(layer_index=0),
                root_children_mpe=model.mpe(layer_index=model.max_layer_index - 1),
            )
        print(f"Finished with seed {seed}.")

