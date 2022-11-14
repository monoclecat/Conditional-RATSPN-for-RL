import logging
from typing import Dict

import math
import numpy as np
import torch as th
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from layers import Product, AbstractLayer
from type_checks import check_valid
from utils import Sample

logger = logging.getLogger(__name__)


class Leaf(AbstractLayer):
    """
    Abstract layer that maps each input feature into a specified
    representation, e.g. Gaussians.

    Implementing layers shall be valid distributions.

    If the input at a specific position is NaN, the variable will be marginalized.
    """

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """
        Create the leaf layer.

        Args:
            in_features: Number of input features.
            out_channels: Number of parallel representations for each input feature.
            num_repetitions: Number of parallel repetitions of this layer.
            dropout: Dropout probability.
        """
        super().__init__(in_features=in_features, num_repetitions=num_repetitions)
        self.out_channels = check_valid(out_channels, int, 1)
        dropout = check_valid(dropout, float, 0.0, 1.0)
        self.dropout = nn.Parameter(th.tensor(dropout), requires_grad=False)

        self.out_shape = f"(N, {in_features}, {out_channels})"

        # Marginalization constant
        self.marginalization_constant = nn.Parameter(th.zeros(1), requires_grad=False)

        # Dropout bernoulli
        self._bernoulli_dist = th.distributions.Bernoulli(probs=self.dropout)

    def _apply_dropout(self, x: th.Tensor) -> th.Tensor:
        # Apply dropout sampled from a bernoulli during training (model.train() has been called)
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            x[dropout_indices] = 0.0
        return x

    def _marginalize_input(self, x: th.Tensor) -> th.Tensor:
        # Marginalize nans set by user
        x = th.where(~th.isnan(x), x, self.marginalization_constant)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_channels={self.out_channels}, dropout={self.dropout}, out_shape={self.out_shape})"


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
        min_sigma: float = 0.0,
        max_sigma: float = 1.0,
        min_mean: float = None,
        max_mean: float = None,
        no_tanh_log_prob_correction: bool = False,
        stds_in_lin_space: bool = True,
        stds_sigmoid_bound: bool = True,
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

        self._tanh_squash = tanh_squash
        self._no_tanh_log_prob_correction = no_tanh_log_prob_correction
        self._stds_in_lin_space = stds_in_lin_space
        self._stds_sigmoid_bound = stds_sigmoid_bound
        self._ratspn = ratspn

        if min_sigma is not None and max_sigma is not None:
            # Init from normal
            self.std_param = nn.Parameter(th.randn(1, in_features, out_channels, num_repetitions))
        else:
            # Init uniform between 0 and 1
            self.std_param = nn.Parameter(th.rand(1, in_features, out_channels, num_repetitions))
        self.min_sigma = check_valid(min_sigma, float, 0.0, max_sigma, allow_none=False)
        self.max_sigma = check_valid(max_sigma, float, min_sigma, allow_none=False)
        self.min_log_sigma = math.log(self.min_sigma + 1e-8)
        self.max_log_sigma = math.log(self.max_sigma + 1e-8)

        if self._tanh_squash:
            assert min_mean is None and max_mean is None, \
                "When leaves are tanh-squashed, the min_mean and max_mean values are predefined and cannot be set."
            min_mean = -6.0
            max_mean = 6.0
        self.min_mean = check_valid(min_mean, float, upper_bound=max_mean, allow_none=True)
        self.max_mean = check_valid(max_mean, float, min_mean, allow_none=True)
        self._means_sigmoid_bound = True

    @property
    def device(self):
        return self.marginalization_constant.device

    def bounded_means(self, means: th.Tensor = None):
        if means is None:
            means = self.mean_param
        if self.min_mean is not None or self.max_mean is not None:
            if self._means_sigmoid_bound:
                mean_ratio = th.sigmoid(means)
                means = self.min_mean + (self.max_mean - self.min_mean) * mean_ratio
            else:
                means = th.clamp(means, self.min_mean, self.max_mean)
        return means

    def bounded_stds(self, stds: th.Tensor = None):
        if stds is None:
            stds = self.std_param
        if self._stds_in_lin_space:
            if self._stds_sigmoid_bound:
                sigma_ratio = th.sigmoid(stds)
                stds = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma_ratio
            else:
                stds = F.softplus(stds) + self.min_sigma
        else:
            stds = F.softplus(stds) + self.min_log_sigma
            # stds = th.clamp(stds, self.min_log_sigma, self.max_log_sigma)
        return stds

    @property
    def means(self):
        if self._ratspn:
            return self.bounded_means()
        else:
            assert ((self.min_mean <= self.mean_param) & (self.mean_param <= self.max_mean)).all()
            return self.mean_param

    @means.setter
    def means(self, means: th.Tensor):
        if isinstance(means, np.ndarray):
            means = th.as_tensor(means)
        means = self.bounded_means(means)
        self.mean_param = nn.Parameter(th.as_tensor(means, dtype=th.float, device=self.device))

    @property
    def log_stds(self):
        if self._ratspn:
            std_param = self.bounded_stds()
        else:
            std_param = self.std_param
        if self._stds_in_lin_space:
            assert ((self.min_sigma <= std_param) & (std_param <= self.max_sigma)).all()
            return std_param.log()
        else:
            return std_param

    @property
    def stds(self):
        if self._ratspn:
            std_param = self.bounded_stds()
        else:
            std_param = self.std_param
        if self._stds_in_lin_space:
            assert ((self.min_sigma <= std_param) & (std_param <= self.max_sigma)).all()
            return std_param
        else:
            return std_param.exp()

    @property
    def var(self):
        return self.stds ** 2

    def set_no_tanh_log_prob_correction(self):
        self._no_tanh_log_prob_correction = False

    def forward(self, x, detach_params: bool = False):
        """
        Forward pass through the leaf layer.

        Args:
            x: th.Tensor of shape
                    [*batch_dims, conditionals, self.config.F, output_channels, self.config.R]
                    batch_dims: Sample shape per conditional.
                    conditionals: In CSPNs, weights are different for each conditional. In RatSpn, this is 1.
                    output_channels: self.config.I or 1 if x should be evaluated on each distribution of a leaf scope
                If tanh squashing of the samples is enabled, x must be from the unsquashed distribution,
                i.e. from the distribution with infinite support!
            layer_index: Evaluate log-likelihood of x at layer
            detach_params: If True, all leaf parameters involved in calculating the log_probs are detached
        Returns:
            th.Tensor: Log-likelihood of the input.
        """

        correction = None
        if self._tanh_squash and not self._no_tanh_log_prob_correction:
            # This correction term assumes that x is from a distribution with infinite support
            correction = 2 * (np.log(2) - x - F.softplus(-2 * x))
            # This correction term assumes the input to be squashed already
            # correction = th.log(1 - x**2 + 1e-6)

        means = self.means.detach() if detach_params else self.means
        stds = self.stds.detach() if detach_params else self.stds
        if x.isnan().any():
            log_std = stds.log()
            var = stds ** 2
            repeat = np.asarray(means.shape[-2:]) // np.asarray(x.shape[-2:])
            x = x.repeat(*([1] * (x.dim()-2)), *repeat)
            mask = ~x.isnan()
            means = means.expand_as(x)
            log_std = log_std.expand_as(x)
            var = var.expand_as(x)
            x[mask] = -((x[mask] - means[mask]) ** 2) / (2 * var[mask]) - log_std[mask] - math.log(math.sqrt(2 * math.pi))
        else:
            d = dist.Normal(means, stds)
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
            # ctx.parent_indices [nr_nodes, *batch_dims, w, self.config.F, r]
            selected_means = means.expand(*ctx.parent_indices.shape[:-1], -1, -1)

            if ctx.repetition_indices is not None:
                rep_ind = ctx.repetition_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                rep_ind = rep_ind.expand(*selected_means.shape[:-1], -1)
                selected_means = th.gather(selected_means, dim=-1, index=rep_ind)

            # Select means and std in the output_channel dimension
            par_ind = ctx.parent_indices.unsqueeze(-2)
            selected_means = th.gather(selected_means, dim=-2, index=par_ind).squeeze(-2)
            selected_means = selected_means.squeeze(-1)
            if not ctx.is_mpe:
                selected_stds = stds.expand(*ctx.parent_indices.shape[:-1], -1, -1)
                if ctx.repetition_indices is not None:
                    selected_stds = th.gather(selected_stds, dim=-1, index=rep_ind)
                selected_stds = th.gather(selected_stds, dim=-2, index=par_ind).squeeze(-2)
                selected_stds = selected_stds.squeeze(-1)

        else:  # mode == 'onehot'
            # ctx.parent_indices shape [nr_nodes, *batch_dims, w, f, oc, r]
            # means shape [w, f, oc, r]
            selected_means = means * ctx.parent_indices
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

    def forward(self, x: th.Tensor, reduction='sum', detach_params: bool = False):
        # Pass through base leaf
        x = self.base_leaf(x, detach_params=detach_params)
        x = self.pad_input(x)

        # Pass through product layer
        x = self.prod(x, reduction=reduction)
        return x

    def entropy(self):
        ent = self.base_leaf.entropy().unsqueeze(0)
        ent = self.pad_input(ent)
        ent = self.prod(ent, reduction='sum')
        return ent

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
