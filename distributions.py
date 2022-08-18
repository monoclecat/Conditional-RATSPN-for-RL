import logging
from typing import Dict, Tuple, List, Optional

import math
import numpy as np
import torch as th
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from base_distributions import Leaf, dist_forward
from layers import Product, Sum
from type_checks import check_valid
from utils import Sample

logger = logging.getLogger(__name__)


class RatNormal(Leaf):
    """ Implementation as in RAT-SPN

    Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(
        self,
        in_features: int,
        out_channels: int,
        ratspn: bool,
        num_repetitions: int,
        dropout: float = 0.0,
        tanh_squash: bool = False,
        min_sigma: float = None,
        max_sigma: float = None,
        min_mean: float = None,
        max_mean: float = None,
        no_tanh_log_prob_correction: bool = False,
    ):
        """Create a gaussian layer.

        Args:
            in_features: Number of input features.
            out_channels: Number of parallel representations for each input feature.
            tanh_bounds: If set, a correction term is applied to the log probs.
            ratspn (bool): If True, dist params will be bounded at each function call.
        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)

        # Create gaussian means and stds
        self.mean_param = nn.Parameter(th.randn(1, in_features, out_channels, num_repetitions))
        self.in_features = in_features
        self.num_repetitions = num_repetitions

        self._tanh_squash = tanh_squash
        self._no_tanh_log_prob_correction = no_tanh_log_prob_correction
        self._ratspn = ratspn

        if min_sigma is not None and max_sigma is not None:
            # Init from normal
            self.log_std_param = nn.Parameter(th.randn(1, in_features, out_channels, num_repetitions))
        else:
            # Init uniform between 0 and 1
            self.log_std_param = nn.Parameter(th.rand(1, in_features, out_channels, num_repetitions))

        self.min_sigma = check_valid(min_sigma, float, 0.0, max_sigma, allow_none=True)
        self.max_sigma = check_valid(max_sigma, float, min_sigma, allow_none=True)
        self.min_mean = check_valid(min_mean, float, upper_bound=max_mean, allow_none=True)
        self.max_mean = check_valid(max_mean, float, min_mean, allow_none=True)

        self._dist_params_are_bounded = False

    def bounded_means(self, means: th.Tensor = None):
        if means is None:
            means = self.mean_param
        if self._tanh_squash:
            means = th.clamp(means, -6.0, 6.0)
        elif self.min_mean is not None or self.max_mean is not None:
            means = th.clamp(means, self.min_mean, self.max_mean)
        return means

    def bounded_log_stds(self, log_stds: th.Tensor = None):
        if log_stds is None:
            log_stds = self.log_std_param
        LOG_STD_MAX = 2
        LOG_STD_MIN = -15
        log_stds = th.clamp(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        return log_stds

    @property
    def means(self):
        if self._ratspn:
            return self.bounded_means()
        else:
            return self.mean_param

    @means.setter
    def means(self, means: th.Tensor):
        if isinstance(means, np.ndarray):
            means = th.as_tensor(means)
        means = self.bounded_means(means)
        self.mean_param = nn.Parameter(th.as_tensor(means, dtype=th.float, device=self.mean_param.device))

    @property
    def log_stds(self):
        if self._ratspn:
            return self.bounded_log_stds()
        else:
            return self.log_std_param

    @log_stds.setter
    def log_stds(self, log_stds: th.Tensor):
        if isinstance(log_stds, np.ndarray):
            log_stds = th.as_tensor(log_stds)
        log_stds = self.bounded_log_stds(log_stds)
        self.log_std_param = nn.Parameter(th.as_tensor(log_stds, dtype=th.float, device=self.log_std_param.device))

    @property
    def stds(self):
        return self.log_stds.exp()

    @property
    def var(self):
        return self.stds ** 2

    def set_no_tanh_log_prob_correction(self):
        self._no_tanh_log_prob_correction = False

    def forward(self, x):
        """
        Forward pass through the leaf layer.

        Args:
            x:
                Input of shape
                    [*batch_dims, weight_sets, self.config.F, output_channels, self.config.R]
                    batch_dims: Sample shape per weight set (= per conditional in the CSPN sense).
                    weight_sets: In CSPNs, weights are different for each conditional. In RatSpn, this is 1.
                    output_channels: self.config.I or 1 if x should be evaluated on each distribution of a leaf scope
            layer_index: Evaluate log-likelihood of x at layer
        Returns:
            th.Tensor: Log-likelihood of the input.
        """

        correction = None
        if self._tanh_squash and not self._no_tanh_log_prob_correction:
            # This correction term assumes that the input is from a distribution with infinite support
            correction = 2 * (np.log(2) - x - F.softplus(-2 * x))
            # This correction term assumes the input to be squashed already
            # correction = th.log(1 - x**2 + 1e-6)

        if x.isnan().any():
            means = self.means
            log_std = self.log_stds
            var = self.var
            repeat = np.asarray(means.shape[-2:]) // np.asarray(x.shape[-2:])
            x = x.repeat(*([1] * (x.dim()-2)), *repeat)
            mask = ~x.isnan()
            means = means.expand_as(x)
            log_std = log_std.expand_as(x)
            var = var.expand_as(x)
            x[mask] = -((x[mask] - means[mask]) ** 2) / (2 * var[mask]) - log_std[mask] - math.log(math.sqrt(2 * math.pi))
        else:
            d = dist.Normal(self.means, self.log_stds.exp())
            x = d.log_prob(x)  # Shape: [n, w, d, oc, r]

        if self._tanh_squash and not self._no_tanh_log_prob_correction:
            x -= correction

        x = self._marginalize_input(x)
        x = self._apply_dropout(x)

        return x

    def sample(self, mode: str = None, ctx: Sample = None):
        """
        Perform sampling, given indices from the parent layer that indicate which of the multiple representations
        for each input shall be used.
        """
        means = self.means
        stds = self.stds
        selected_means, selected_stds, rep_ind = None, None, None

        if ctx.is_root:
            selected_means = means.expand(*ctx.n, -1, -1, -1, -1)
            if not ctx.is_mpe:
                selected_stds = stds.expand(*ctx.n, -1, -1, -1, -1)
        elif mode == 'index':
            w, d, i, r = means.shape
            # ctx.parent_indices [nr_nodes, *batch_dims, w, self.config.F, r]
            selected_means = means.expand(*ctx.parent_indices.shape[:-3], *means.shape)

            if ctx.repetition_indices is not None:
                rep_ind = ctx.repetition_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                rep_ind = rep_ind.expand(*selected_means.shape[:-1], -1)
                selected_means = th.gather(selected_means, dim=-1, index=rep_ind)

            # Select means and std in the output_channel dimension
            par_ind = ctx.parent_indices.unsqueeze(-2)
            selected_means = th.gather(selected_means, dim=-2, index=par_ind).squeeze(-2)
            selected_means = selected_means.squeeze(-1)
            if not ctx.is_mpe:
                selected_stds = stds.expand(*ctx.parent_indices.shape[:-3], *stds.shape)
                if ctx.repetition_indices is not None:
                    selected_stds = th.gather(selected_stds, dim=-1, index=rep_ind)
                selected_stds = th.gather(selected_stds, dim=-2, index=par_ind).squeeze(-2)
                selected_stds = selected_stds.squeeze(-1)

        else:  # mode == 'onehot'
            # ctx.parent_indices shape [nr_nodes, *batch_dims, w, f, oc, r]
            # means shape [w, f, oc, r]
            selected_means = means * ctx.parent_indices
            assert ctx.parent_indices.detach().sum(-2).max().item() == 1.0
            selected_means = selected_means.sum(-2)
            if not ctx.has_rep_dim:
                # Only one repetition is selected, remove repetition dim of parameters
                selected_means = selected_means.sum(-1)

            if not ctx.is_mpe:
                selected_stds = stds * ctx.parent_indices
                selected_stds = selected_stds.sum(-2)
                if not ctx.has_rep_dim:
                    # Only one repetition is selected, remove repetition dim of parameters
                    selected_stds = selected_stds.sum(-1)

        if ctx.is_mpe:
            samples = selected_means
        else:
            gauss = dist.Normal(selected_means, selected_stds)
            samples = gauss.rsample()
        return samples

    def sample_index_style(self, ctx: Sample = None) -> th.Tensor:
        return self.sample(mode='index', ctx=ctx)

    def sample_onehot_style(self, ctx: Sample = None) -> th.Tensor:
        return self.sample(mode='onehot', ctx=ctx)

    def _get_base_distribution(self) -> th.distributions.Distribution:
        gauss = dist.Normal(self.means, self.stds)
        return gauss


class IndependentMultivariate(Leaf):
    def __init__(
        self,
        in_features: int,
        out_channels: int,
        cardinality: int,
        ratspn: bool,
        num_repetitions: int,
        dropout: float = 0.0,
        tanh_squash: bool = False,
        leaf_base_class: Leaf = RatNormal,
        leaf_base_kwargs: Dict = None,
    ):
        """
        Create multivariate distribution that only has non zero values in the covariance matrix on the diagonal.

        Args:
            out_channels: Number of parallel representations for each input feature.
            cardinality: Number of variables per gauss.
            in_features: Number of input features.
            dropout: Dropout probabilities.
            leaf_base_class (Leaf): The encapsulating base leaf layer class.
            ratspn (bool): If True, dist params will be bounded at each function call.
        """
        super(IndependentMultivariate, self).__init__(in_features, out_channels, num_repetitions, dropout)
        if leaf_base_kwargs is None:
            leaf_base_kwargs = {}

        self.base_leaf = leaf_base_class(
            out_channels=out_channels,
            in_features=in_features,
            dropout=dropout,
            num_repetitions=num_repetitions,
            tanh_squash=tanh_squash,
            ratspn=ratspn,
            **leaf_base_kwargs,
        )
        self._pad = (cardinality - self.in_features % cardinality) % cardinality
        # Number of input features for the product needs to be extended depending on the padding applied here
        prod_in_features = in_features + self._pad
        self.prod = Product(
            in_features=prod_in_features, cardinality=cardinality, num_repetitions=num_repetitions
        )

        self.cardinality = check_valid(cardinality, int, 1, in_features + 1)
        self.out_shape = f"(N, {self.prod._out_features}, {out_channels}, {self.num_repetitions})"

    @property
    def out_features(self):
        return self.prod._out_features

    def _init_weights(self):
        if isinstance(self.base_leaf, RatNormal):
            truncated_normal_(self.base_leaf.stds, std=0.5)

    def pad_input(self, x: th.Tensor):
        if self._pad:
            x = F.pad(x, pad=[0, 0, 0, 0, 0, self._pad], mode="constant", value=0.0)
        return x

    @property
    def pad(self):
        return self._pad

    def forward(self, x: th.Tensor, reduction='sum'):
        # Pass through base leaf
        x = self.base_leaf(x)
        x = self.pad_input(x)

        # Pass through product layer
        x = self.prod(x, reduction=reduction)
        return x

    def entropy(self):
        ent = self.base_leaf.entropy().unsqueeze(0)
        ent = self.pad_input(ent)
        ent = self.prod(ent, reduction='sum')
        return ent

    def _get_base_distribution(self):
        raise Exception("IndependentMultivariate does not have an explicit PyTorch base distribution.")

    def sample(self, mode: str = None, ctx: Sample = None):
        if not ctx.is_root:
            ctx = self.prod.sample(ctx=ctx)

            # Remove padding
            if self._pad:
                ctx.parent_indices = ctx.parent_indices[..., :-self._pad, :]

        samples = self.base_leaf.sample(ctx=ctx, mode=mode)
        return samples

    def sample_index_style(self, ctx: Sample = None) -> th.Tensor:
        return self.sample(ctx=ctx, mode='index')

    def sample_onehot_style(self, ctx: Sample = None) -> th.Tensor:
        return self.sample(ctx=ctx, mode='onehot')

    def __repr__(self):
        return f"IndependentMultivariate(in_features={self.in_features}, out_channels={self.out_channels}, dropout={self.dropout}, cardinality={self.cardinality}, out_shape={self.out_shape})"


class GaussianMixture(IndependentMultivariate):
    def __init__(
            self,
            in_features: int,
            gmm_modes: int,
            out_channels: int,
            cardinality: int,
            num_repetitions: int = 1,
            dropout: float = 0.0,
            leaf_base_class: Leaf = RatNormal,
            leaf_base_kwargs: Dict = None,
    ):
        super(GaussianMixture, self).__init__(
            in_features=in_features, out_channels=gmm_modes, cardinality=cardinality, num_repetitions=num_repetitions,
            dropout=dropout, leaf_base_class=leaf_base_class, leaf_base_kwargs=leaf_base_kwargs)
        self.sum = Sum(in_features=self.out_features, in_channels=gmm_modes, num_repetitions=num_repetitions,
                       out_channels=out_channels, dropout=dropout)

    def forward(self, x: th.Tensor, reduction='sum'):
        x = super().forward(x=x, reduction=reduction)
        if reduction is None:
            x = self._weighted_sum(x)
            return x
        else:
            return self.sum(x)

    def sample(self, ctx: Sample = None) -> th.Tensor:
        context_overhang = ctx.parent_indices.size(1) - self.sum.in_features
        assert context_overhang >= 0, f"context_overhang is negative! ({context_overhang})"
        if context_overhang:
            ctx.parent_indices = ctx.parent_indices[:, : -context_overhang]
        ctx = self.sum.sample(ctx=ctx)
        return super(GaussianMixture, self).sample(ctx=ctx)

    def _weighted_sum(self, x: th.Tensor):
        weights = self.sum.weights
        if weights.dim() == 5:
            # Only in the Cspn case are the weights already log-normalized
            weights = weights.exp()
        else:
            weights = F.softmax(weights, dim=2)

        # weights.unsqueeze(2) is of shape [n, d, 1, ic, oc, r]
        # The extra dimension is created so all elements of the gradient vectors are multiplied by the same
        # weight for that feature and output channel.
        return (x.unsqueeze(4) * weights.unsqueeze(2)).sum(dim=3)

    def iterative_gmm_entropy_lb(self, reduction='mean'):
        """
            Calculate the entropy lower bound of the first-level mixtures.
            See "On Entropy Approximation for Gaussian Mixture Random Vectors" Huber et al. 2008, Theorem 2
        """
        log_gmm_weights: th.Tensor = self.sum.weights
        if log_gmm_weights.dim() == 4:
            # Only in the Cspn case are the weights already log-normalized
            log_gmm_weights = th.log_softmax(log_gmm_weights, dim=2)
        assert self.sum.weights.dim() == 5, "This isn't adopted to the 4-dimensional RatSpn weights yet"
        N, D, I, S, R = log_gmm_weights.shape
        # First sum layer after the leaves has weights of dim (N, D, I, S, R)
        # The entropy lower bound must be calculated for every sum node

        # bounded mean and variance
        # dist weights are of size (N, F, I, R)
        means, var = self.base_leaf.moments()
        _, F, _, _ = means.shape

        lb_ent_i = []
        for i in range(I):
            log_probs_i = []
            for j in range(I):
                summed_vars = var[:, :, [i], :] + var[:, :, [j], :]
                component_log_probs = -((means[:, :, [i], :] - means[:, :, [j], :]) ** 2) / (2 * summed_vars) - \
                                      0.5 * summed_vars.log() - math.log(math.sqrt(2 * math.pi))
                component_log_probs = self.prod(self.pad_input(component_log_probs))

                # Unsqueeze in output channel dimension so that the log_prob vector of each feature
                # is added to the weights of the S sum nodes of that feature.
                component_log_probs.unsqueeze_(dim=3)
                log_probs_i.append(log_gmm_weights[:, :, [j], :, :] + component_log_probs)
            log_probs_i = th.cat(log_probs_i, dim=2).logsumexp(dim=2, keepdim=True)
            lb_ent_i.append(log_gmm_weights[:, :, [i], :, :].exp() * log_probs_i)
        lb_ent = -th.cat(lb_ent_i, dim=2).sum(dim=2)

        if reduction == 'mean':
            lb_ent = lb_ent.mean()
        return lb_ent

    def gmm_entropy_lb(self, reduction='mean'):
        """
            Calculate the entropy lower bound of the first-level mixtures.
            See "On Entropy Approximation for Gaussian Mixture Random Vectors" Huber et al. 2008, Theorem 2
        """
        log_gmm_weights: th.Tensor = self.sum.weights
        if log_gmm_weights.dim() == 4:
            # Only in the Cspn case are the weights already log-normalized
            log_gmm_weights = th.log_softmax(log_gmm_weights, dim=2)
        assert self.sum.weights.dim() == 5, "This isn't adopted to the 4-dimensional RatSpn weights yet"
        N, D, I, S, R = log_gmm_weights.shape
        # First sum layer after the leaves has weights of dim (N, D, I, S, R)
        # The entropy lower bound must be calculated for every sum node

        # bounded mean and variance
        # dist weights are of size (N, F, I, R)
        means, var = self.base_leaf.moments()
        _, F, _, _ = means.shape

        repeated_means = means.repeat(1, 1, I, 1)
        var_outer_sum = var.unsqueeze(2) + var.unsqueeze(3)
        var_outer_sum = var_outer_sum.view(N, F, I**2, R)
        means_to_eval = means.repeat_interleave(I, dim=2)

        component_log_probs = -((means_to_eval - repeated_means) ** 2) / (2 * var_outer_sum) - \
                              0.5 * var_outer_sum.log() - math.log(math.sqrt(2 * math.pi))
        log_probs = self.prod(self.pad_input(component_log_probs))

        # Match sum weights to log probs
        w_j = log_gmm_weights.repeat(1, 1, I, 1, 1)
        # now [N, D, I^2, S, R]

        # Unsqueeze in output channel dimension so that the log_prob vector of each RV is added to the weights of
        # the S sum nodes of that RV.
        log_probs.unsqueeze_(dim=3)

        weighted_log_probs = w_j + log_probs
        lb_log_term = th.logsumexp(weighted_log_probs.view(N, D, I, I, S, R), dim=2)
        # lb_log_term is now [N, D, I, S, R], the same shape as log_gmm_weights

        gmm_ent_lb = -(log_gmm_weights.exp() * lb_log_term)
        gmm_ent_lb = gmm_ent_lb.sum(dim=2).sum(dim=1)
        # The entropies of all features can be summed up => [N, OC, R]
        if reduction == 'mean':
            gmm_ent_lb = gmm_ent_lb.mean()
        return gmm_ent_lb

    def __repr__(self):
        return f"GaussianMixture(in_features={self.in_features}, out_channels={self.out_channels}, dropout={self.dropout}, cardinality={self.cardinality}, out_shape={self.out_shape})"


def truncated_normal_(tensor, mean=0, std=0.1):
    """
    Truncated normal from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
