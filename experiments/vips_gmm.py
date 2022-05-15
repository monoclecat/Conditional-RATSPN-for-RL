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
import gif
import os
import time

def build_ratspn(F, I):
    config = RatSpnConfig()
    config.C = 1
    config.F = F
    config.R = 2
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


@gif.frame
def gif_frame(probs):
    fig, (ax1) = plt.subplots(1, figsize=(10, 10), dpi=100)
    ax1.imshow(probs.detach().exp().view(grid_spacing, grid_spacing).cpu().numpy())
    ax1.set_title(f"RatSpn distribution at step {step}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='../../spn_experiments',
                        help='The base directory where the directory containing the results will be saved to.')
    parser.add_argument('--resp_with_grad', action='store_true',
                        help="If True, approximation of responsibilities is done with grad enabled.")
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--fit_to_prison', action='store_true',
                        help="If True, the SPN is trained to fit a wall-devided cell rectangle cell structure. ")
    args = parser.parse_args()

    th.manual_seed(args.seed)

    for d in [args.results_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    num_dimensions = 2
    if args.fit_to_prison:
        target_mixture = Target(num_dimensions)
    else:
        num_true_components = 10
        target_gmm_prior_variance = 1e3
        [target_lnpdf, _, _, target_mixture] = build_GMM_lnpdf(num_dimensions, num_true_components,
                                                               target_gmm_prior_variance)

    grid_spacing = 501
    x = np.linspace(-50, 50, grid_spacing)
    grid = np.stack(np.meshgrid(x, x), axis=-1)
    grid = grid.reshape(-1, 2)

    target_probs = target_mixture.evaluate(grid).reshape(grid_spacing, grid_spacing)
    if True:
        plt.imshow(target_probs)
        plt.title(f"Target distribution {f'with {num_true_components} components' if not args.fit_to_prison else ''}")
        plt.show()
    model = build_ratspn(
        num_dimensions,
        15 if args.fit_to_prison else int(num_true_components * 1.0)
    ).to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    grid_tensor = th.as_tensor(grid, device=model.device, dtype=th.float)

    fps = 10
    gif_duration = 10  # seconds
    if args.fit_to_prison:
        n_steps = 1000
    else:
        n_steps = 48000 if args.resp_with_grad else 3000
    n_frames = fps * gif_duration

    def bookmark():
        pass
    frames = []
    losses = []
    t_start = time.time()
    for step in range(int(n_steps)):
        if step % int(n_steps / n_frames) == 0:
            probs = model(grid_tensor)
            frame = gif_frame(probs)
            frames.append(frame)
            t_delta = np.around(time.time() - t_start, 2)
            print(f"Time delta: {time_delta(t_delta)} - Avg. loss at step {step}: {round(np.mean(losses), 2)}")
            losses = []
            t_start = time.time()

        if args.fit_to_prison:
            sample = model.sample(mode='onehot', n=100)
            log_prob = model(sample)
            loss = -target_mixture.evaluate(sample) * log_prob
        else:
            ent, _ = model.vi_entropy_approx(sample_size=25, grad_thru_resp=args.resp_with_grad)
            loss = -ent.mean() * 10.0
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    save_path = os.path.join(
        args.results_dir,
        f"{'grad_enabled' if args.resp_with_grad else 'no_grad'}_seed{args.seed}.gif"
    )
    gif.save(frames, save_path, duration=1/fps, unit='s')

    print(1)

