import logging
from abc import ABC, abstractmethod
from typing import List, Union, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as dist

from type_checks import check_valid
from utils import SamplingContext

logger = logging.getLogger(__name__)


class AbstractLayer(nn.Module, ABC):
    def __init__(self, in_features: int, num_repetitions: int = 1):
        super().__init__()
        self.in_features = check_valid(in_features, int, 1)
        self.num_repetitions = check_valid(num_repetitions, int, 1)

    @abstractmethod
    def sample(self, context: SamplingContext = None) -> Union[SamplingContext, torch.Tensor]:
        """
        Sample from this layer.
        Args:
            context: Sampling context.

        Returns:
            torch.Tensor: Generated samples.
        """
        pass


class Sum(AbstractLayer):
    def __init__(
        self, in_channels: int, in_features: int, out_channels: int, num_repetitions: int = 1, dropout: float = 0.0
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
        """
        super().__init__(in_features, num_repetitions)

        self.in_channels = check_valid(in_channels, int, 1)
        self.out_channels = check_valid(out_channels, int, 1)
        self.dropout = nn.Parameter(torch.tensor(check_valid(dropout, float, 0.0, 1.0)), requires_grad=False)

        # Weights, such that each sumnode has its own weights
        ws = torch.randn(self.in_features, self.in_channels, self.out_channels, self.num_repetitions)
        self.weights = nn.Parameter(ws)
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=self.dropout)

        self.out_shape = f"(N, {self.in_features}, {self.out_channels}, {self.num_repetitions})"

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_input_cache_enabled = False
        self._input_cache = None

        # Weights of this sum node that were propagated down through the following product layer.
        # These weights weigh the child sum node in the layer after the product layer that follows this one.
        # The consolidated weights are needed for moment calculation.
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
        return self.weights.device

    def forward(self, x: torch.Tensor):
        """
        Sum layer forward pass.

        Args:
            x: Input of shape [batch, weight_sets, in_features, in_channels].
                weight_sets: In CSPNs, there are separate weights for each batch element.

        Returns:
            torch.Tensor: Output of shape [batch, in_features, out_channels]
        """
        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache = x.clone()

        # Apply dropout: Set random sum node children to 0 (-inf in log domain)
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            x[dropout_indices] = np.NINF

        # Dimensions
        n, w, d, ic, r = x.size()
        x = x.unsqueeze(4)  # Shape: [n, w, d, ic, 1, r]
        if self.weights.dim() == 4:
            # RatSpns only have one set of weights, so we must augment the weight_set dimension
            weights = self.weights.unsqueeze(0)
        else:
            weights = self.weights
        # Weights is of shape [n, d, ic, oc, r]
        oc = weights.size(3)
        # The weights must be expanded by the batch dimension so all samples of one conditional see the same weights.
        log_weights = weights.unsqueeze(0)

        # Multiply (add in log-space) input features and weights
        x = x + log_weights  # Shape: [n, w, d, ic, oc, r]

        # Compute sum via logsumexp along in_channels dimension
        x = torch.logsumexp(x, dim=3)  # Shape: [n, w, d, oc, r]

        # Assert correct dimensions
        assert x.size() == (n, w, d, oc, r)

        return x

    def sample(self, context: SamplingContext = None) -> SamplingContext:
        """Method to sample from this layer, based on the parents output.

        Output is always a vector of indices into the channels.

        Args:
            context: Contains
                repetition_indices (List[int]): An index into the repetition axis for each sample.
                    Can be None if `num_repetitions==1`.
                parent_indices (torch.Tensor): Parent sampling output.
                n: Number of samples to draw for each set of weights.
        Returns:
            torch.Tensor: Index into tensor which paths should be followed.
        """

        # Sum weights are of shape: [N, D, IC, OC, R]
        # We now want to use `indices` to access one in_channel for each in_feature x out_channels block
        # index is of size in_feature
        weights: torch.Tensor = self.weights.data
        if weights.dim() == 4:
            weights = weights.unsqueeze(0)
        # w is the number of weight sets
        w, d, ic, oc, r = weights.shape
        sample_size = context.n

        # Create sampling context if this is a root layer
        if context.is_root:
            if False:
                # Initialize rep indices
                context.repetition_indices = torch.zeros(sample_size, w, dtype=int, device=self.__device)
                weights = weights[:, :, :, 0, 0].unsqueeze(0).repeat(sample_size, 1, 1, 1)

                # Select weights, repeat bs times along the last dimension
                # weights = weights[:, :, [0] * w, 0]  # Shape: [D, IC, N]

                # Move sample dimension to the first axis: [feat, channels, batch] -> [batch, feat, channels]
                # weights = weights.permute(2, 0, 1)  # Shape: [N, D, IC]
            else:
                weights = weights.unsqueeze(0).expand(sample_size, -1, -1, -1, -1, -1)
                # weights from selected repetition with shape [n, w, d, ic, oc, r]
                # In this sum layer there are oc * r nodes per feature. oc * r is our nr_nodes.
                weights = weights.permute(5, 4, 0, 1, 2, 3)
                # weights from selected repetition with shape [r, oc, n, w, d, ic]
                # Reshape weights to [oc * r, n, w, d, ic]
                # The nodes in the first dimension are taken from the first two weight dimensions [r, oc] like this:
                # [0, 0], ..., [0, oc-1], [1, 0], ..., [1, oc-1], [2, 0], ..., [r-1, oc-1]
                # This means the weights for the first oc nodes are the weights for repetition 0.
                # This must coincide with the repetition indices.
                weights = weights.reshape(oc * r, sample_size, w, d, ic)

                context.repetition_indices = torch.arange(r).to(self.__device).repeat_interleave(oc)
                context.repetition_indices = context.repetition_indices.unsqueeze(-1).unsqueeze(-1).repeat(
                    1, context.n, w
                )
        else:
            # If this is not the root node, use the paths (out channels), specified by the parent layer
            self._check_repetition_indices(context)

            # Iterative version that might be easier to understand. The iterative version is a lot slower.
            # Iterative: timeit 1000 iterations: 9.53s on CPU, 20.4s on GPU
            # Tensorized: timeit 1000 iterations: 1.85s on CPU, 0.44s on GPU
            # node_selected_weights = []
            # for n in range(context.parent_indices.size(0)):
            #     # Iterate over the sampling contexts of each node
            #     parent_selected_weights = []
            #     for i in range(sample_size):
            #         # Each node is sampled sample_size times
            #         # Select the weights at the requested repetitions. This eliminates the rep dimension of weights
            #         selected_weights = weights[range(w), :, :, :, context.repetition_indices[n, i]]
            #         # weights from selected rep with shape [w, d, ic, oc]
            #         parent_indices = context.parent_indices[n, i].unsqueeze(-1).expand(-1, -1, ic).unsqueeze(-1)
            #         # parent_indices [w, d, ic, 1]
            #         selected_weights = selected_weights.gather(dim=-1, index=parent_indices).squeeze(-1)
            #         # weights from selected parent with shape [w, d, ic]
            #
            #         parent_selected_weights.append(selected_weights)
            #     node_selected_weights.append(torch.stack(parent_selected_weights, dim=0))

            weights = weights.unsqueeze(0).unsqueeze(0)
            rep_ind = context.repetition_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            weights = weights.expand(rep_ind.shape[0], sample_size, -1, -1, -1, -1, -1)
            rep_ind = rep_ind.expand(-1, -1, -1, d, ic, oc, -1)
            weights = torch.gather(weights, dim=-1, index=rep_ind).squeeze(-1)
            # weights from selected repetition with shape [nr_nodes, n, w, d, ic, oc]
            parent_indices = context.parent_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, ic, -1)
            weights = torch.gather(weights, dim=-1, index=parent_indices).squeeze(-1)
            # weights from selected parent with shape [nr_nodes, n, w, d, ic]

        # Check dimensions
        assert weights.shape[1:] == (sample_size, w, d, ic)  # [n, w, d, ic]

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and self._input_cache is not None:
            raise NotImplementedError("Not yet adapted to new sampling method")
            for i in range(w):
                # Reweight the i-th samples weights by its likelihood values at the correct repetition
                weights[i, :, :] += self._input_cache[i, :, :, context.repetition_indices[i]]

        # If sampling context is MPE, set max weight to 1 and rest to zero, such that the maximum index will be sampled
        if context.is_mpe:
            # Get index of largest weight along in-channel dimension
            indices = weights.argmax(dim=-1)
        else:
            # Create categorical distribution and use weights as logits.
            #
            # Use the Gumble-Softmax trick to obtain one-hot indices of the categorical distribution
            # represented by the given logits. (Use Gumble-Softmax instead of Categorical
            # to allow for gradients).
            #
            # The code below is an approximation of:
            #
            # >> dist = torch.distributions.Categorical(logits=weights)
            # >> indices = dist.sample()

            cats = torch.arange(ic, device=weights.device)
            one_hot = F.gumbel_softmax(logits=weights, hard=True, dim=-1)
            indices = (one_hot * cats).sum(-1).long()

        # Update parent indices
        context.parent_indices = indices

        return context

    def depr_forward_grad(self, child_grads):
        weights = self.consolidated_weights.unsqueeze(2)
        return [(g.unsqueeze_(4) * weights).sum(dim=3) for g in child_grads]

    def depr_compute_moments(self, child_moments: List[torch.Tensor]):
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

    def _check_repetition_indices(self, context: SamplingContext):
        assert context.repetition_indices.shape[0] == context.parent_indices.shape[0]
        assert context.repetition_indices.shape[1] == context.parent_indices.shape[1]
        if self.num_repetitions > 1 and context.repetition_indices is None:
            raise Exception(
                f"Sum layer has multiple repetitions (num_repetitions=={self.num_repetitions}) but repetition_indices argument was None, expected a Long tensor size #samples."
            )
        if self.num_repetitions == 1 and context.repetition_indices is None:
            context.repetition_indices = torch.zeros(context.n, dtype=int, device=self.__device)

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
        # self._conv_weights = nn.Parameter(torch.ones(1, 1, cardinality, 1, 1), requires_grad=False)
        self._pad = (self.cardinality - self.in_features % self.cardinality) % self.cardinality

        self._out_features = np.ceil(self.in_features / self.cardinality).astype(int)
        self.out_shape = f"(N, {self._out_features}, in_channels, {self.num_repetitions})"

    # @property
    # def __device(self):
        # """Hack to obtain the current device, this layer lives on."""
        # return self._conv_weights.device

    def forward(self, x: torch.Tensor, reduction = 'sum'):
        """
        Product layer forward pass.

        Args:
            x: Input of shape [batch, weight_sets, in_features, channel, repetitions].

        Returns:
            torch.Tensor: Output of shape [batch, ceil(in_features/cardinality), channel].
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
        n, w, d, c, r = x.size()
        d_out = d // self.cardinality

        if reduction is None:
            return x.view(n, w, d_out, self.cardinality, c, r)
        elif reduction == 'sum':
            batch_result = []
            for i in range(n):
                # Use convolution with 3D weight tensor filled with ones to add the log-probs together
                x_batch = x[i].unsqueeze(1)  # Shape: [w, 1, d, c, r]
                result = F.conv3d(x_batch,
                                  weight=torch.ones(1, 1, self.cardinality, 1, 1, device=x.device),
                                  stride=(self.cardinality, 1, 1))

                # Remove simulated channel
                result = result.squeeze(1)
                assert result.size() == (w, d_out, c, r)
                batch_result.append(result)
            return torch.stack(batch_result)
        else:
            assert False, "No reduction other than sum is implemented. "

    def sample(self, n: int = None, context: SamplingContext = None) -> SamplingContext:
        """Method to sample from this layer, based on the parents output.

        Args:
            n (int): Number of instances to sample.
            indices (torch.Tensor): Parent sampling output.
        Returns:
            torch.Tensor: Index into tensor which paths should be followed.
                          Output should be of size: in_features, out_channels.
        """

        # If this is a root node
        if context.is_root:

            if self.num_repetitions == 1:
                # If there is only a single repetition, create new sampling context
                context.parent_indices = torch.zeros(context.n, 1, dtype=int, device=self.__device)
                context.repetition_indices = torch.zeros(context.n, dtype=int, device=self.__device)
                return context
            else:
                raise Exception(
                    "Cannot start sampling from Product layer with num_repetitions > 1 and no context given."
                )
        else:
            # Repeat the parent indices, e.g. [0, 2, 3] -> [0, 0, 2, 2, 3, 3] depending on the cardinality
            indices = torch.repeat_interleave(context.parent_indices, repeats=self.cardinality, dim=3)

            # Remove padding
            if self._pad:
                indices = indices[:, :, :, : -self._pad]

            context.parent_indices = indices
            return context

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
        self._pad = 2 ** np.ceil(np.log2(in_features)).astype(np.int) - in_features
        super().__init__(2 ** np.ceil(np.log2(in_features)).astype(np.int), num_repetitions)

        self.in_channels = check_valid(in_channels, int, 1)
        cardinality = 2  # Fixed to binary graphs for now
        self.cardinality = check_valid(cardinality, int, 2, in_features + 1)
        self._out_features = np.ceil(self.in_features / self.cardinality).astype(int)

        # Collect scopes for each product child
        self._scopes = [[] for _ in range(self.cardinality)]

        # Create sequence of scopes
        scopes = np.arange(self.in_features)

        # For two consecutive scopes
        for i in range(0, self.in_features, self.cardinality):
            for j in range(cardinality):
                self._scopes[j].append(scopes[i + j])
                # if i + j < in_features:
                    # self._scopes[j].append(scopes[i + j])
                # else:
                    # Case: d mod cardinality != 0 => Create marginalized nodes with prob 1.0
                    # Pad x in forward pass on the right: [n, d, c] -> [n, d+1, c] where index
                    # d+1 is the marginalized node (index "in_features")
                    # self._scopes[j].append(self.in_features)

        # Transform into numpy array for easier indexing
        self._scopes = np.array(self._scopes)

        # Create index map from flattened to coordinates (only needed in sampling)
        self.unraveled_channel_indices = nn.Parameter(
            torch.tensor([(i, j) for i in range(self.in_channels)
                          for j in range(self.in_channels)]),
            requires_grad=False
        )
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

    def forward(self, x: torch.Tensor):
        """
        Product layer forward pass.

        Args:
            x: Input of shape [batch, weight_sets, in_features, channel].
                weight_sets: In CSPNs, there are separate weights for each batch element.

        Returns:
            torch.Tensor: Output of shape [batch, ceil(in_features/2), channel * channel].
        """
        if self._pad:
            # Pad marginalized node
            x = F.pad(x, pad=[0, 0, 0, 0, 0, self._pad], mode="constant", value=0.0)

        # Dimensions
        n, w, d, c, r = x.size()
        d_out = d // self.cardinality

        # Build outer sum, using broadcasting, this can be done with
        # modifying the tensor dimensions:
        # left: [n, d/2, c, r] -> [n, d/2, c, 1, r]
        # right: [n, d/2, c, r] -> [n, d/2, 1, c, r]
        left = x[:, :, self._scopes[0, :], :, :].unsqueeze(4)
        right = x[:, :, self._scopes[1, :], :, :].unsqueeze(3)

        # left + right with broadcasting: [n, d/2, c, 1, r] + [n, d/2, 1, c, r] -> [n, d/2, c, c, r]
        result = left + right

        # Put the two channel dimensions from the outer sum into one single dimension:
        # [n, d/2, c, c, r] -> [n, d/2, c * c, r]
        result = result.view(n, w, d_out, c * c, r)

        assert result.size() == (n, w, d_out, c * c, r)
        return result

    def sample(self, context: SamplingContext = None) -> SamplingContext:
        """Method to sample from this layer, based on the parents output.

        Args:
            context (SamplingContext):
                n: Number of samples.
                parent_indices (torch.Tensor): Nodes selected by parent layer
                repetition_indices (torch.Tensor): Repetitions selected by parent layer
        Returns:
            torch.Tensor: Index into tensor which paths should be followed.
                          Output should be of size: in_features, out_channels.
        """

        # If this is a root node
        if context.is_root:
            # Sampling starting at a CrossProduct layer means sampling each node in the layer.

            # Thus, there are oc*r*out_features sets of SamplingContexts.
            # oc*r of the nodes are represented by the size of the first
            # dimension in parent_indices and repetition indices.
            # The nodes over all 'out_features', for a given oc and r, are collected in the
            # last dimension of parent_indices.
            # The first product nodes across all features are in parent_indices[0,0,0,:]
            # The individual nodes per in_feature can be taken out of parent_indices by splitting it in half
            # in the last dimension.

            # The parent and repetition indices are also repeated by the number of samples requested
            # and by the number of conditionals the CSPN weights have been set to.
            # In the RatSpn case, the number of conditionals (abbreviated by w) is 1.
            indices = self.unraveled_channel_indices.data.unsqueeze(1).unsqueeze(1)
            # indices is [oc, 1, 1, cardinality]
            indices = indices.repeat(
                self.num_repetitions, context.n, self.num_conditionals, self.in_features//self.cardinality
            )
            # indices is [oc*r, n, w, in_features]
            oc, _ = self.unraveled_channel_indices.shape
            # Repetition indices say for which repetition(s) the parent_indices should be applied
            context.repetition_indices = torch.arange(self.num_repetitions).to(self.__device).repeat_interleave(oc)
            context.repetition_indices = context.repetition_indices.unsqueeze(-1).unsqueeze(-1).repeat(
                1, context.n, self.num_conditionals
            )
        else:
            nr_nodes, n, w, d = context.parent_indices.shape
            # Map flattened indexes back to coordinates to obtain the chosen input_channel for each feature
            indices = self.unraveled_channel_indices[context.parent_indices].to(context.parent_indices.device)
            indices = indices.view(nr_nodes, n, w, -1)

        # Remove padding
        if self._pad:
            indices = indices[:, :, : -self._pad]

        context.parent_indices = indices
        return context

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
        parent_weights = torch.stack((left_sums_weights, right_sums_weights), dim=2)
        parent_weights = parent_weights.view(n, self.in_features, self.in_channels, p_oc, r)
        return parent_weights

    def __repr__(self):
        return "CrossProduct(in_features={}, out_shape={})".format(self.in_features, self.out_shape)
