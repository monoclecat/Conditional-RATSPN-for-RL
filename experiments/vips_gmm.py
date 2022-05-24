import numpy as np
from scipy.stats import multivariate_normal as normal_pdf
from scipy import log, exp
import torch as th
import matplotlib.pyplot as plt
from rat_spn import RatSpn, RatSpnConfig
from distributions import RatNormal
from experiments.train_mnist import count_params
from experiments.train_cspn_mnist_gen import time_delta
from torch import optim
import torch.distributions as dist
import gif
import os
import time

def build_ratspn(F, I):
    config = RatSpnConfig()
    config.C = 1
    config.F = F
    config.R = 3
    config.D = int(np.log2(F))
    config.I = I
    config.S = 3
    config.dropout = 0.0
    config.leaf_base_class = RatNormal
    model = RatSpn(config)
    count_params(model)
    return model


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
            x: Shape [ic, s, n, w, self.config.F, self.config.R]
                Samples of the SPN to evaluate the target log-probs of
            dims:
            return_log:

        Returns:

        """
        if dims is None:
            dims = [i for i in range(self.num_dimensions)]

        if has_rep_dim:
            x = th.einsum('...ij -> ...ji', x)

        if True and probs is not None:
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
            log_probs = th.stack(log_probs, dim=-2)
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
    parser.add_argument('--seed', '-s', type=int, nargs='+', required=True)
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='cuda or cpu')
    parser.add_argument('--ent_approx_sample_size', '-samples', type=int, default=5)
    parser.add_argument('--results_dir', type=str, default='../../spn_experiments',
                        help='The base directory where the directory containing the results will be saved to.')
    parser.add_argument('--resp_with_grad', action='store_true',
                        help="If True, approximation of responsibilities is done with grad enabled.")
    parser.add_argument('--make_gif', '-gif', action='store_true',
                        help="Create gif of plots")
    args = parser.parse_args()

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
        dist_params = {
            0: {
                'mean': th.as_tensor([-40, -30, -20, -10, 10, 20, 30, 40]),
                'std': th.as_tensor([1] * 8),
            },
            1: {
                'mean': th.as_tensor([-35, -15, 15, 35]),
                'std': th.as_tensor([3] * 4)
            }
        }
        for dim, params in dist_params.items():
            target_mixture.add_dim(params['mean'], params['std'])

    grid_points = 501
    min_x = -50
    max_x = -min_x
    x = th.linspace(min_x, max_x, grid_points)
    grid = th.stack(th.meshgrid(x, x), dim=-1)
    grid = grid.reshape(-1, 2)

    def target_callback(x):
        return target_mixture.evaluate(x, dims=None, return_log=True, has_rep_dim=True)

    def exp_view(tensor: th.Tensor):
        return tensor.exp().view(grid_points, grid_points).T

    def forplot(tensor: th.Tensor, scale=False):
        tensor = tensor.detach().cpu().numpy()
        if scale:
            tensor = (tensor - min_x)/(max_x - min_x) * grid_points
        return tensor

    @gif.frame
    def gif_frame(probs):
        fig, (ax1) = plt.subplots(1, figsize=(10, 10), dpi=100)
        ax1.imshow(forplot(exp_view(probs)))
        ax1.set_title(f"RatSpn distribution at step {step}")


    target_probs = exp_view(target_mixture.evaluate(grid, dims=None, return_log=True))
    if isinstance(target_mixture, IndGmm):
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 15), dpi=100)
        ax1.imshow(target_probs)
        ax1.set_title(
            f"Target distribution with "
            f"{', '.join([f'{target_mixture.num_components[i]} components over {dim}' for i, dim in enumerate(['x', 'y'])])}."
        )
        ax2.plot([i for i in range(grid_points)], forplot(target_probs.exp().sum(0), True))
        ax2.set_title("Marginal target distribution over x")
        ax3.plot([i for i in range(grid_points)], forplot(target_probs.exp().sum(1), True))
        ax3.set_title("Marginal target distribution over y")
        plt.show()
    else:
        plt.imshow(target_probs)
        plt.title(f"Target distribution with {num_true_components} components.")
        plt.show()

    for seed in args.seed:
        th.manual_seed(seed)

        load_path = os.path.join(args.results_dir, 'high_ent_ratspn.pt')
        if load_path is None:
            model = build_ratspn(
                num_dimensions,
                int(num_true_components * 1.0)
            ).to(args.device)
        else:
            model = th.load(load_path, map_location=args.device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        grid_tensor = th.as_tensor(grid, device=model.device, dtype=th.float)

        if args.make_gif:
            fps = 10
            gif_duration = 10  # seconds
            n_steps = 48000 if args.resp_with_grad else 3000
            n_frames = fps * gif_duration
            make_frame_every = int(n_steps / n_frames)
        else:
            n_steps = 50000
            make_frame_every = 1 # 5000

        def bookmark():
            pass
        frames = []
        losses = []
        t_start = time.time()
        for step in range(int(n_steps)):
            if step % make_frame_every == 0:
                probs = model(grid_tensor)
                if args.make_gif:
                    frame = gif_frame(probs)
                    frames.append(frame)
                else:
                    plt.imshow(forplot(exp_view(probs)))
                    plt.show()
                t_delta = np.around(time.time() - t_start, 2)
                print(f"Time delta: {time_delta(t_delta)} - Avg. loss at step {step}: {round(np.mean(losses), 2)}")
                losses = []
                t_start = time.time()

            if True:
                child_entropies = None
                for layer_index in range(model.num_layers):
                    child_entropies, layer_log, _ = model.layer_entropy_approx(
                        layer_index=layer_index, child_entropies=child_entropies,
                        sample_size=9, grad_thru_resp=False, verbose=False,
                        target_dist_callback=target_callback,
                        return_child_samples=True,
                    )

            if False:
                child_entropies = None
                logging = {}
                for layer_index in range(1):
                    if False:
                        temp = model._leaf.base_leaf.means.detach()
                        temp[:, 0, :, :] = -25
                        del model._leaf.base_leaf.mean_param
                        model._leaf.base_leaf.mean_param = temp
                        temp = model._leaf.base_leaf.stds.detach()
                        temp[:, 0, :, :] = 1e-3
                        del model._leaf.base_leaf.std_param
                        model._leaf.base_leaf.std_param = temp
                        probs = model(grid_tensor)

                    child_entropies, layer_log, samples = model.layer_entropy_approx(
                        layer_index=layer_index, child_entropies=child_entropies,
                        sample_size=1,
                        grad_thru_resp=False, verbose=True,
                        return_child_samples=True,
                        sample_post_processing_kwargs={'split_by_scope': True},
                    )
                    samples = samples.permute(0, 5, 1, 2, 3, 4)
                    # samples [ic, r, s, n, w, f]
                    samples = samples[0, 0]
                    # samples = th.as_tensor([[5.7994, th.nan], [th.nan, -5.8928]], device=args.device).\
                        # unsqueeze(1).unsqueeze(1).expand_as(samples)

                    flat_samples = samples.flatten(0, -2)
                    nan_samples = flat_samples.isnan()
                    x_samples = flat_samples[:, 0][~nan_samples[:, 0]]
                    y_samples = flat_samples[:, 1][~nan_samples[:, 1]]

                    filled_samples = model.sample(mode='index', n=5000, evidence=samples.unsqueeze(-1)).sample
                    filled_samples = filled_samples.squeeze(-1).squeeze(0)
                    flat_filled = filled_samples.flatten(1, -2)
                    x_filled_in = flat_filled[:, nan_samples[:, 0]]
                    y_filled_in = flat_filled[:, nan_samples[:, 1]]

                    if True:
                        # All samples where x is given and y must be filled in
                        fig, (ax1) = plt.subplots(1, figsize=(15, 15), dpi=300)
                        ax1.imshow(forplot(exp_view(probs)))
                        ax1.vlines(forplot(x_samples, True), ymin=0, ymax=grid_points-1, color='r', alpha=0.3)
                        plt.plot(forplot(y_filled_in[..., 0].flatten(), True),
                                 forplot(y_filled_in[..., 1].flatten(), True),
                                 marker='.', color='r', linestyle='', label='(x, y) ~ p(y | x)')
                        plt.show()

                    if True:
                        # One sample where x is given and y must be filled in
                        sample_id = 0
                        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 15), dpi=300)
                        ax1.imshow(forplot(exp_view(probs)))
                        ax1.vlines(forplot(x_samples[sample_id], True), ymin=0, ymax=grid_points-1, color='r', alpha=0.3)
                        ax1.plot(forplot(y_filled_in[:, sample_id, 0].flatten(), True),
                                 forplot(y_filled_in[:, sample_id, 1].flatten(), True),
                                 marker='.', color='r', linestyle='', label='(x, y) ~ p(y | x)')

                        x_coord = np.round(forplot(y_filled_in[0, sample_id, 0].flatten(), True)).astype(int).item()
                        ax2.plot([i for i in range(grid_points)], forplot(exp_view(probs)[:, x_coord].squeeze()))
                        sampled_y = forplot(y_filled_in[:, sample_id, 1], True)
                        ax3.hist(sampled_y, bins=int(grid_points/2))
                        plt.show()

                    if True:
                        fig, (ax1) = plt.subplots(1, figsize=(15, 15), dpi=300)
                        ax1.imshow(forplot(exp_view(probs)))
                        ax1.hlines(forplot(y_samples, True), xmin=0, xmax=grid_points-1, linestyles=':', alpha=0.7)
                        plt.legend()
                        plt.show()

                    filled_samples = filled_samples.permute(0, 2, 3, 4, 5, 1)
                    samples = samples.squeeze(-1).permute(0, 2, 3, 4, 5, 1)
                    logging.update(layer_log)

            if False:
                model.sample(mode='index', n=(3, 7))
                ent, _ = model.vi_entropy_approx(
                    sample_size=args.ent_approx_sample_size,
                    grad_thru_resp=args.resp_with_grad
                )
                loss = -ent.mean() * 10.0
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if args.make_gif:
            save_path = os.path.join(
                args.results_dir,
                f"{'grad_enabled' if args.resp_with_grad else 'no_grad'}"
                f"_samples{args.ent_approx_sample_size}"
                f"_seed{seed}.gif"
            )
            gif.save(frames, save_path, duration=1/fps, unit='s')

        print(f"Finished with seed {seed}")

