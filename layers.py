import logging
from abc import ABC, abstractmethod
from typing import List, Union, Tuple

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
import torch.distributions as dist

from type_checks import check_valid
from utils import Sample

logger = logging.getLogger(__name__)


class AbstractLayer(nn.Module, ABC):
    def __init__(self, in_features: int, num_repetitions: int = 1):
        super().__init__()
        self.in_features = check_valid(in_features, int, 1)
        self.num_repetitions = check_valid(num_repetitions, int, 1)

    @abstractmethod
    def sample(self, ctx: Sample = None) -> Union[Sample, th.Tensor]:
        """
        Sample from this layer.
        Args:
            ctx: Sampling context.

        Returns:
            th.Tensor: Generated samples.
        """
        pass

    @abstractmethod
    def sample_index_style(self, ctx: Sample = None) -> Union[Sample, th.Tensor]:
        pass

    @abstractmethod
    def sample_onehot_style(self, ctx: Sample = None) -> Union[Sample, th.Tensor]:
        pass


class Sum(AbstractLayer):
    def __init__(
        self, in_channels: int, in_features: int, out_channels: int, ratspn: bool, num_repetitions: int,
            dropout: float = 0.0,
    ):
        """
        Create a Sum layer.

        Input is expected to be of shape [n, d, ic, r].
        Output will be of shape [n, d, oc, r].

        Args:
            in_channels (int): Number of output channels from the previous layer.
            in_features (int): Number of input features.
            out_channels (int): Multiplicity of a sum node for a given scope set.
            num_repetitions(int): Number of layer repetitions in parallel.
            dropout (float, optional): Dropout percentage.
            ratspn (bool): If True, weights will be log-normalized at each function call.
        """
        super().__init__(in_features, num_repetitions)

        self.in_channels = check_valid(in_channels, int, 1)
        self.out_channels = check_valid(out_channels, int, 1)
        self.dropout = nn.Parameter(th.tensor(check_valid(dropout, float, 0.0, 1.0)), requires_grad=False)
        self.ratspn = ratspn

        # Weights, such that each sumnode has its own weights. weight_sets := w = 1 in the RatSpn case.
        ws = th.randn(1, self.in_features, self.in_channels, self.out_channels, self.num_repetitions)
        self.weight_param = nn.Parameter(ws)
        self._bernoulli_dist = th.distributions.Bernoulli(probs=self.dropout)

        self.out_shape = f"(N, {self.in_features}, {self.out_channels}, {self.num_repetitions})"

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_input_cache_enabled = False
        self._input_cache = None

        # Weights of this sum node that were propagated down through the following product layer.
        # These weights weigh the child sum node in the layer after the product layer that follows this one.
        self.consolidated_weights = None

        self.mean = None
        self.var = None
        self.skew = None

    def _enable_input_cache(self):
        """Enables the input cache. This will store the input in forward passes into `self.__input_cache`."""
        self._is_input_cache_enabled = True

    def _disable_input_cache(self):
        """Disables and clears the input cache."""
        self._is_input_cache_enabled = False
        self._input_cache = None

    @property
    def __device(self):
        """Hack to obtain the current device, this layer lives on."""
        return self.weight_param.device

    @property
    def weights(self):
        # weights need to have shape [w, d, ic, oc, r]. In the RatSpn case, w is always 1
        if self.ratspn:
            return F.log_softmax(self.weight_param, dim=2)
        else:
            return self.weight_param

    def forward(self, x: th.Tensor, detach_params: bool = False):
        """
        Sum layer forward pass.

        Args:
            x: Input of shape [batch, weight_sets, in_features, in_channels, repetitions].
                weight_sets: In CSPNs, there are separate weights for each batch element.

        Returns:
            th.Tensor: Output of shape [batch, in_features, out_channels]
        """
        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache = x.detach().clone()

        # Apply dropout: Set random sum node children to 0 (-inf in log domain)
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            x[dropout_indices] = np.NINF

        # Dimensions
        w, d, c, r = x.shape[-4:]
        batch_dims = x.shape[:-4]
        x = x.unsqueeze(-2)  # Shape: [*batch_dims, w, d, ic, 1, r]
        weights: th.Tensor = self.weights.detach() if detach_params else self.weights

        # Weights is of shape [w, d, ic, oc, r]
        oc = weights.size(3)

        # Multiply (add in log-space) input features and weights
        x = x + weights  # Shape: [n, w, d, ic, oc, r]

        # Compute sum via logsumexp along in_channels dimension
        x = th.logsumexp(x, dim=-3)  # Shape: [n, w, d, oc, r]

        # Assert correct dimensions
        assert x.size() == (*batch_dims, w, d, oc, r)

        return x

    def sample(self, mode: str = None, ctx: Sample = None) -> Union[Sample, th.Tensor]:
        """Method to sample from this layer, based on the parents output.

        Output is always a vector of indices into the channels.

        Args:
            mode: Two sampling modes are supported:
                'index': Sampling mechanism with indexes, which are non-differentiable.
                'onehot': This sampling mechanism work with one-hot vectors, grouped into tensors.
                          This way of sampling is differentiable, but also takes almost twice as long.
            ctx: Contains
                repetition_indices (List[int]): An index into the repetition axis for each sample.
                    Can be None if `num_repetitions==1`.
                parent_indices (th.Tensor): Parent sampling output.
                n: Number of samples to draw for each set of weights.
        Returns:
            th.Tensor: Index into tensor which paths should be followed.
        """
        assert mode is not None, "A sampling mode must be provided!"
        clear_weights_mask = None

        # Sum weights are of shape: [N, D, IC, OC, R]
        # We now want to use `indices` to access one in_channel for each in_feature x out_channels block
        # index is of size in_feature
        weights: th.Tensor = self.weights
        # w is the number of weight sets
        w, d, ic, oc, r = weights.shape
        sample_shape = ctx.n

        # Create sampling context if this is a root layer
        if ctx.is_root:
            weights = weights.repeat(*sample_shape, *([1] * weights.dim()))
            # weights from selected repetition with shape [*batch_dims, w, d, ic, oc, r]
            weights = th.einsum('...ijk -> j...ik', weights)
            # oc is our nr_nodes: [nr_nodes=oc, *batch_dims, w, d, ic, r]
        else:
            if mode == 'index':
                # If this is not the root node, use the paths (out channels), specified by the parent layer
                # parent_indices [nr_nodes, *sample_shape, w, d, r]
                # weights [w, d, ic, oc, r]
                weights = weights.expand(*ctx.parent_indices.shape[:-1], -1, -1, -1)
                if ctx.repetition_indices is not None:
                    rep_ind = ctx.repetition_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    rep_ind = rep_ind.expand(*weights.shape[:-1], -1)
                    weights = th.gather(weights, dim=-1, index=rep_ind)
                    # weights from selected repetition with shape [nr_nodes, n, w, d, ic, oc, 1]

                parent_indices = ctx.parent_indices.unsqueeze(-2).expand(*weights.shape[:-2], -1)
                # parent_indices [nr_nodes, *sample_shape, w, d, ic, r]
                weights = th.gather(weights, dim=-2, index=parent_indices.unsqueeze(-2)).squeeze(-2)
                # weights from selected parent with shape [nr_nodes, *sample_shape, w, d, ic, r]
            else:  # mode == 'onehot'
                # parent_indices [nr_nodes, *sample_shape, w, d, oc, r]
                weights = weights * ctx.parent_indices.unsqueeze(-3)
                # [nr_nodes, *sample_shape, w, d, ic, oc, r]
                weights = weights.sum(-2)
                clear_weights_mask = (ctx.parent_indices.sum(-2, keepdim=True) == 0.0).expand_as(weights)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and self._input_cache is not None:
            weight_offsets = self._input_cache.clone()
            if mode == 'index':
                if ctx.repetition_indices is not None:
                    rep_ind = ctx.repetition_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    rep_ind = rep_ind.expand(*([-1] * ctx.repetition_indices.dim()), d, ic, -1)
                    weight_offsets = weight_offsets.expand(*rep_ind.shape[:-1], -1)
                    # Both are now [nr_nodes, *sample_shape, w, d, ic, r]
                    weight_offsets = th.gather(weight_offsets, dim=-1, index=rep_ind)
            else:  # mode == 'onehot'
                if ctx.parent_indices is not None:
                    # ctx.parent_indices is missing the ic dim, inp_cache is missing the oc dim
                    weight_offsets = weight_offsets.unsqueeze(-2) * ctx.parent_indices.unsqueeze(-3)
                    weight_offsets = weight_offsets.sum(-2)
            weights = weights + weight_offsets

        # If sampling context is MPE, set max weight to 1 and rest to zero, such that the maximum index will be sampled
        if ctx.is_mpe:
            # Get index of largest weight along in-channel dimension
            indices = weights.argmax(dim=-2)
            # indices [nr_nodes, *sample_shape, w, d, r]
            if mode == 'onehot':
                # Get index of largest weight along in-channel dimension
                one_hot = F.one_hot(indices, num_classes=ic)
                # indices [nr_nodes, *sample_shape, w, d, r, ic]
                indices = th.einsum('...ij -> ...ji', one_hot)
        else:
            # Create categorical distribution and use weights as logits.
            #
            # Use the Gumble-Softmax trick to obtain one-hot indices of the categorical distribution
            # represented by the given logits. (Use Gumble-Softmax instead of Categorical
            # to allow for gradients).
            #
            # The code below is an approximation of:
            #
            # >> dist = th.distributions.Categorical(logits=weights)
            # >> indices = dist.sample()

            indices = F.gumbel_softmax(logits=weights, hard=True, dim=-2)  # -2 == ic (input channel) dimension
            if mode == 'index':
                cats = th.arange(ic, device=weights.device)
                cats = cats.unsqueeze_(-1).expand(-1, indices.size(-1))
                indices = (indices * cats).sum(-2).long()

        if mode == 'onehot' and clear_weights_mask is not None:
            indices[clear_weights_mask] = 0.0

        ctx.parent_indices = indices
        return ctx

    def sample_index_style(self, ctx: Sample = None) -> Sample:
        return self.sample(mode='index', ctx=ctx)

    def sample_onehot_style(self, ctx: Sample = None) -> Sample:
        return self.sample(mode='onehot', ctx=ctx)

    def depr_forward_grad(self, child_grads):
        weights = self.consolidated_weights.unsqueeze(2)
        return [(g.unsqueeze_(4) * weights).sum(dim=3) for g in child_grads]

    def depr_compute_moments(self, child_moments: List[th.Tensor]):
        assert self.consolidated_weights is not None, "No consolidated weights are set for this Sum node!"
        # Create an extra dimension for the mean vector so all elements of the mean vector are multiplied by the same
        # weight for that feature and output channel.
        weights = self.consolidated_weights.unsqueeze(2)
        # Weights is of shape [n, d, 1, ic, oc, r]

        mean, var, skew = [m.unsqueeze(4) if m is not None else None for m in child_moments]
        # moments have shape [n, d, cardinality, ic, r]
        # Create an extra 'output channels' dimension, as the weights are separate for each output channel.
        self._mean = mean * weights
        # _mean has shape [n, d, cardinality, ic, oc, r]
        self._mean = self._mean.sum(dim=3)
        # _mean has shape [n, d, cardinality, oc, r]

        centered_mean = mean - self._mean.unsqueeze(4)
        self._var = var + centered_mean**2
        self._var = self._var * weights
        self._var = self._var.sum(dim=3)

        self._skew = 3*centered_mean*var + centered_mean**3
        if skew is not None:
            self._skew = self._skew + skew
        self._skew = self._skew * weights
        self._skew = self._skew.sum(dim=3)

        return self._mean, self._var, self._skew

    def __repr__(self):
        return "Sum(in_channels={}, in_features={}, out_channels={}, dropout={}, out_shape={})".format(
            self.in_channels, self.in_features, self.out_channels, self.dropout, self.out_shape
        )


class Product(AbstractLayer):
    """
    Product Node Layer that chooses k scopes as children for a product node.
    """

    def __init__(self, in_features: int, cardinality: int, num_repetitions: int = 1):
        """
        Create a product node layer.

        Args:
            in_features (int): Number of input features.
            cardinality (int): Number of random children for each product node.
        """

        super().__init__(in_features, num_repetitions)

        self.cardinality = check_valid(cardinality, int, 1, in_features + 1)

        # Implement product as convolution
        # self._conv_weights = nn.Parameter(th.ones(1, 1, cardinality, 1, 1), requires_grad=False)
        self._pad = (self.cardinality - self.in_features % self.cardinality) % self.cardinality

        self._out_features = np.ceil(self.in_features / self.cardinality).astype(int)
        self.out_shape = f"(N, {self._out_features}, in_channels, {self.num_repetitions})"

    # @property
    # def __device(self):
        # """Hack to obtain the current device, this layer lives on."""
        # return self._conv_weights.device

    def forward(self, x: th.Tensor, reduction = 'sum', **kwargs):
        """
        Product layer forward pass.

        Args:
            x: Input of shape [*batch_dims, weight_sets, in_features, channel, repetitions].

        Returns:
            th.Tensor: Output of shape [batch, ceil(in_features/cardinality), channel].
        """
        # Only one product node
        if self.cardinality == x.shape[2]:
            if reduction == 'sum':
                return x.sum(2, keepdim=True)
            else:
                return x

        # Special case: if cardinality is 1 (one child per product node), this is a no-op
        if self.cardinality == 1:
            return x

        # Pad if in_features % cardinality != 0
        if self._pad > 0:
            x = F.pad(x, pad=(0, 0, 0, 0, 0, self._pad), value=0)

        # Dimensions
        w, d, c, r = x.shape[-4:]
        n = x.shape[:-4]  # Any number of batch dimensions
        d_out = d // self.cardinality
        x = x.view(*n, w, d_out, self.cardinality, c, r)

        if reduction is None:
            return x
        elif reduction == 'sum':
            return x.sum(dim=-3)
        else:
            raise NotImplementedError("No reduction other than sum is implemented. ")

    def sample_onehot_style(self, ctx: Sample = None) -> Union[Sample, th.Tensor]:
        return self.sample(ctx)

    def sample_index_style(self, ctx: Sample = None) -> Union[Sample, th.Tensor]:
        return self.sample(ctx)

    def sample(self, n: int = None, ctx: Sample = None) -> Sample:
        """Method to sample from this layer, based on the parents output.

        Args:
            n (int): Number of instances to sample.
            indices (th.Tensor): Parent sampling output.
        Returns:
            th.Tensor: Index into tensor which paths should be followed.
        """

        # If this is a root node
        if ctx.is_root:

            if self.num_repetitions == 1:
                # If there is only a single repetition, create new sampling context
                ctx.parent_indices = th.zeros(*ctx.n, 1, dtype=int, device=self.__device)
                ctx.repetition_indices = th.zeros(*ctx.n, dtype=int, device=self.__device)
                return ctx
            else:
                raise Exception(
                    "Cannot start sampling from Product layer with num_repetitions > 1 and no context given."
                )
        else:
            # Repeat the parent indices, e.g. [0, 2, 3] -> [0, 0, 2, 2, 3, 3] depending on the cardinality
            ctx.parent_indices = self.repeat_by_cardinality(
                ctx.parent_indices, -2 if ctx.sampling_mode == 'index' else -3
            )
            return ctx

    def repeat_by_cardinality(self, x, feature_dim=-2):
        """
        Repeat the feature dimension of x by the leaf cardinality and remove padding.

        Args:
            x: th.Tensor of shape [batch_dims and others, d, r] in the case that feature_dim == -2
            feature_dim: Feature dimension which will be repeated
        """
        x = th.repeat_interleave(x, repeats=self.cardinality, dim=feature_dim)

        # Remove padding
        if self._pad:
            x, _ = th.tensor_split(x, [-self._pad], dim=feature_dim)
        return x

    def __repr__(self):
        return "Product(in_features={}, cardinality={}, out_shape={})".format(
            self.in_features, self.cardinality, self.out_shape
        )


class CrossProduct(AbstractLayer):
    """
    Layerwise implementation of a RAT Product node.

    Builds the the combination of all children in two regions:
    res = []
    for n1 in R1, n2 in R2:
        res += [n1 * n2]

    TODO: Generalize to k regions (cardinality = k).
    """

    def __init__(self, in_features: int, in_channels: int, num_repetitions: int = 1):
        """
        Create a rat product node layer.

        Args:
            in_features (int): Number of input features.
            in_channels (int): Number of input channels. This is only needed for the sampling pass.
        """

        # Check if padding to next power of 2 is necessary
        self._pad = 2 ** np.ceil(np.log2(in_features)).astype(int) - in_features
        super().__init__(2 ** np.ceil(np.log2(in_features)).astype(np.int), num_repetitions)

        self.in_channels = check_valid(in_channels, int, 1)
        cardinality = 2  # Fixed to binary graphs for now
        self.cardinality = check_valid(cardinality, int, 2, in_features + 1)
        self._out_features = np.ceil(self.in_features / self.cardinality).astype(int)

        # Create index map from flattened to coordinates (only needed in sampling)
        self.unraveled_channel_indices = nn.Parameter(
            th.tensor([(i, j) for i in range(self.in_channels)
                          for j in range(self.in_channels)]),
            requires_grad=False
        )
        self.one_hot_in_channel_mapping = nn.Parameter(F.one_hot(self.unraveled_channel_indices).float(),
                                                       requires_grad=False)
        # Number of conditionals (= number of different weight sets) in the CSPN.
        # This is only needed when sampling this layer as root.
        # It is initialized as 1, which would also be the RatSpn case.
        # It is only set in the CSPN.set_weights() function.
        self.num_conditionals = 1

        self.out_shape = f"(N, {self._out_features}, {self.in_channels ** 2}, {self.num_repetitions})"

    @property
    def __device(self):
        """Hack to obtain the current device, this layer lives on."""
        return self.unraveled_channel_indices.device

    def forward(self, x: th.Tensor, **kwargs):
        """
        Product layer forward pass.

        Args:
            x: Input of shape [any number of batch dims, weight_sets, in_features, in_channels, repetition].
                weight_sets: In CSPNs, there are separate weights for each batch element.

        Returns:
            th.Tensor: Output of shape [batch, ceil(in_features/2), in_channels^2].
        """
        # split_shuffled_scopes will pad x to make the features divisible by the cardinality
        left_scope, right_scope = CrossProduct.split_shuffled_scopes(x, scope_dim=-3, cardinality=self.cardinality)
        left_scope = left_scope.unsqueeze(-2)
        right_scope = right_scope.unsqueeze(-3)
        # left + right with broadcasting: [*n, w, d/2, c, 1, r] + [*n, w, d/2, 1, c, r] -> [*n, w, d/2, c, c, r]
        result = left_scope + right_scope

        # Put the two channel dimensions from the outer sum into one single dimension:
        # [*n, w, d/2, c, c, r] -> [*n, w, d/2, c * c, r]
        result = result.view(*result.shape[:-3], result.shape[-3] * result.shape[-2], result.shape[-1])
        assert result.shape[-3:] == (self.in_features // self.cardinality, self.in_channels ** 2, self.num_repetitions)
        return result

    def sample(self, mode: str = None, ctx: Sample = None) -> Union[Sample, th.Tensor]:
        """Method to sample from this layer, based on the parents output.

        Args:
            ctx (Sample):
                n: Number of samples.
                parent_indices (th.Tensor): Nodes selected by parent layer
                repetition_indices (th.Tensor): Repetitions selected by parent layer
        Returns:
            th.Tensor: Index into tensor which paths should be followed.
                          Output should be of size: in_features, out_channels.
        """
        assert mode is not None, "A sampling mode must be provided!"

        # If this is a root node
        if ctx.is_root:
            # Sampling starting at a CrossProduct layer means sampling each node in the layer.

            # There are oc * r * out_features nodes in this layer.
            # We sample across all repetitions at the same time, so the parent_indices tensor has a repetition dim.

            # The parent and repetition indices are also repeated by the number of samples requested
            # and by the number of conditionals the CSPN weights have been set to.
            # In the RatSpn case, the number of conditionals (abbreviated by w) is 1.
            if mode == 'index':
                indices = self.unraveled_channel_indices.data.unsqueeze(1).unsqueeze(-1)
                for _ in range(len(ctx.n)):
                    indices = indices.unsqueeze(1)
                # indices is [nr_nodes=oc, 1, 1, cardinality, 1]
                indices = indices.repeat(
                    1, *ctx.n, self.num_conditionals, self.in_features//self.cardinality, self.num_repetitions
                )
                # indices is [nr_nodes=oc, n, w, in_features, r]
            else:  # mode == 'onehot'
                indices = self.one_hot_in_channel_mapping.data.unsqueeze(1).unsqueeze(-1)
                for _ in range(len(ctx.n)):
                    indices = indices.unsqueeze(1)
                # indices is [nr_nodes=oc, 1, 1, cardinality, in_channels, 1]
                indices = indices.repeat(
                    1, *ctx.n, self.num_conditionals, self.in_features // self.cardinality, 1, self.num_repetitions
                )
                # indices is [nr_nodes=oc, n, w, in_features, in_channels, r]

            oc, _ = self.unraveled_channel_indices.shape

            # repetition indices are left empty because they are implicitly given in parent_indices
        else:
            if mode == 'index':
                # Map flattened indexes back to coordinates to obtain the chosen input_channel for each feature
                indices = self.unraveled_channel_indices[ctx.parent_indices]
                indices = th.einsum('...ij -> ...ji', indices)
                indices = indices.reshape(*indices.shape[:-3], indices.shape[-3] * self.cardinality, indices.shape[-1])
            else:  # mode == 'onehot'
                indices = th.einsum('...ij -> ...ji', ctx.parent_indices).unsqueeze(-1).unsqueeze(-1)
                indices = indices * self.one_hot_in_channel_mapping
                # Shape [nr_nodes, *batch_dims, w, d, r, oc, 2, ic]
                indices = indices.sum(dim=-3)
                # Shape [nr_nodes, *batch_dims, w, d, r, 2, ic]
                indices = th.einsum('...ijk -> ...jki', indices)
                # [nr_nodes, *batch_dims, w, d, 2, ic, r]
                indices = indices.reshape(*indices.shape[:-4], indices.shape[-4] * self.cardinality, *indices.shape[-2:])
                # [nr_nodes, *batch_dims, w, d * 2, ic, r]

        # Remove padding
        if self._pad:
            # indices is [nr_nodes=oc, n, w, in_features, r]
            indices, _ = th.tensor_split(indices, [-self._pad], dim=-2 if mode == 'index' else -3)

        ctx.parent_indices = indices
        return ctx

    def sample_index_style(self, ctx: Sample = None) -> Sample:
        return self.sample(mode='index', ctx=ctx)

    def sample_onehot_style(self, ctx: Sample = None) -> Sample:
        return self.sample(mode='onehot', ctx=ctx)

    def consolidate_weights(self, parent_weights):
        """
            This function takes the weights from the parent sum nodes meant for this product layer and recalculates
            them as if they would directly weigh the child sum nodes of this product layer.
            This turns the sum-prod-sum chain into a hierarchical mixture: sum-sum.
        """
        n, d, p_ic, p_oc, r = parent_weights.shape
        assert p_ic == self.in_channels**2, \
            "Number of parent input channels isn't the squared input channels of this product layer!"
        assert d*2 == self.in_features, \
            "Number of input features isn't double the output features of this product layer!"
        parent_weights = parent_weights.softmax(dim=2)
        parent_weights = parent_weights.view(n, d, self.in_channels, self.in_channels, p_oc, r)
        left_sums_weights = parent_weights.sum(dim=3)
        right_sums_weights = parent_weights.sum(dim=2)
        # left_sums_weights contains the consolidated weights of each parent's in_feature regarding the left sets of
        # sum nodes for that feature. right_sums_weights analogously for the right sets.
        # We can't simply concatenate them along the 1. dimension because this would mix up the order of in_features
        # of this layer. Along dim 1, we need left[0], right[0], left[1], right[1] => we need to interleave them
        parent_weights = th.stack((left_sums_weights, right_sums_weights), dim=2)
        parent_weights = parent_weights.view(n, self.in_features, self.in_channels, p_oc, r)
        return parent_weights

    def __repr__(self):
        return "CrossProduct(in_features={}, out_shape={})".format(self.in_features, self.out_shape)

    @staticmethod
    def split_shuffled_scopes(t: th.Tensor, scope_dim: int, cardinality: int = 2):
        """
        The scopes in the tensor are split into a left scope and a right scope, where
        the scope dimension contains the shuffled scopes of the left side and the right side,
        [x_{0,L}, x_{0,R}, x_{1,L}, x_{1,R}, x_{2,L}, x_{2,R}, ...]
        """
        shape_before_scope = t.shape[:scope_dim]
        shape_after_scope = t.shape[scope_dim + 1:]
        if padding := t.size(scope_dim) % cardinality != 0:
            pad = th.zeros(*shape_before_scope, padding, *shape_after_scope, device=t.device)
            t = th.cat((t, pad), dim=scope_dim)
        left_scope_right_scope = t.view(
            *shape_before_scope, t.size(scope_dim) // cardinality, cardinality, *shape_after_scope
        )
        index_dim = scope_dim + 1 if scope_dim > 0 else scope_dim
        left_scope, right_scope = th.split(left_scope_right_scope, 1, dim=index_dim)
        left_scope = left_scope.squeeze(index_dim)
        right_scope = right_scope.squeeze(index_dim)
        return left_scope, right_scope

