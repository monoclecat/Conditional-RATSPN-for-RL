import logging
from typing import Dict, Type, List, Union, Optional, Tuple, Callable
import math

import numpy as np
import torch as th
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from torch import distributions as dist
import scipy.optimize
import matplotlib.pyplot as plt

from base_distributions import Leaf
from layers import CrossProduct, Sum
from type_checks import check_valid
from utils import *
from distributions import IndependentMultivariate, GaussianMixture, truncated_normal_

logger = logging.getLogger(__name__)


def invert_permutation(p: th.Tensor):
    """
    The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    Taken from: https://stackoverflow.com/a/25535723, adapted to PyTorch.
    """
    s = th.empty(p.shape[0], dtype=p.dtype, device=p.device)
    s[p] = th.arange(p.shape[0]).to(p.device)
    return s


@dataclass
class RatSpnConfig:
    """
    Class for keeping the RatSpn config. Parameter names are according to the original RatSpn paper.

    in_features: int  # Number of input features
    D: int  # Tree depth
    S: int  # Number of sum nodes at each layer
    I: int  # Number of distributions for each scope at the leaf layer
    R: int  # Number of repetitions
    C: int  # Number of root heads / Number of classes
    dropout: float  # Dropout probabilities for leafs and sum layers
    leaf_base_class: Type  # Type of the leaf base class (Normal, Bernoulli, etc)
    leaf_base_kwargs: Dict  # Parameters for the leaf base class, such as
    #                   tanh_factor: float  # If set, tanh will be applied to samples and taken times this factor.
    gmm_leaves: bool  # If true, the leaves are Gaussian mixtures
    """

    is_ratspn: bool = True
    in_features: int = None
    D: int = None
    S: int = None
    I: int = None
    R: int = None
    C: int = None
    dropout: float = None
    leaf_base_class: Type = None
    leaf_base_kwargs: Dict = None
    gmm_leaves: bool = False
    tanh_squash: bool = None

    @property
    def F(self):
        """Alias for in_features."""
        return self.in_features

    @F.setter
    def F(self, in_features):
        """Alias for in_features."""
        self.in_features = in_features

    def assert_valid(self):
        """Check whether the configuration is valid."""
        self.F = check_valid(self.F, int, 1)
        self.D = check_valid(self.D, int, 1, allow_none=True)
        self.C = check_valid(self.C, int, 1)
        self.S = check_valid(self.S, int, 1)
        self.R = check_valid(self.R, int, 1)
        self.I = check_valid(self.I, int, 1)
        self.dropout = check_valid(self.dropout, float, 0.0, 1.0)
        assert self.leaf_base_class is not None, Exception("RatSpnConfig.leaf_base_class parameter was not set!")
        assert isinstance(self.leaf_base_class, type) and issubclass(
            self.leaf_base_class, Leaf
        ), f"Parameter RatSpnConfig.leaf_base_class must be a subclass type of Leaf but was {self.leaf_base_class}."

        if 2 ** self.D > self.F:
            raise Exception(f"The tree depth D={self.D} must be <= {np.floor(np.log2(self.F))} (log2(in_features).")

    def __setattr__(self, key, value):
        if hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"RatSpnConfig object has no attribute {key}")


class RatSpn(nn.Module):
    """
    RAT SPN PyTorch implementation with layer-wise tensors.

    See also:
    https://arxiv.org/abs/1806.01910
    """
    _inner_layers: nn.ModuleList

    def __init__(self, config: RatSpnConfig):
        """
        Create a RatSpn based on a configuration object.

        Args:
            config (RatSpnConfig): RatSpn configuration object.
        """
        super().__init__()
        config.assert_valid()
        assert config.R != config.F, "The number of repetitions can't be the number of input features to the SPN, sorry."
        self.config = config

        # Construct the architecture
        self._build()

        # Initialize weights
        if self.config.is_ratspn:
            self._init_weights()

        # Obtain permutation indices
        self._make_random_repetition_permutation_indices()

        self.__max_layer_index = len(self._inner_layers) + 1
        self.num_layers = self.__max_layer_index + 1

    @property
    def dtype(self):
        return self._leaf.base_leaf.marginalization_constant.dtype

    @property
    def max_layer_index(self):
        return self.__max_layer_index

    @property
    def sum_layer_indices(self):
        return [i for i in range(2, self.max_layer_index+1, 2)]

    def layer_index_to_obj(self, layer_index):
        if layer_index == 0:
            return self._leaf
        elif layer_index < self.__max_layer_index:
            return self._inner_layers[layer_index-1]
        elif layer_index == self.__max_layer_index:
            return self.root
        else:
            raise IndexError(f"layer_index must take a value between 0 and {self.max_layer_index}, "
                             f"but it was {layer_index}!")

    def _make_random_repetition_permutation_indices(self):
        """Create random permutation indices for each repetition."""
        permutation = []
        inv_permutation = []
        for r in range(self.config.R):
            permutation.append(th.tensor(np.random.permutation(self.config.F)))
            inv_permutation.append(invert_permutation(permutation[-1]))
        # self.permutation: th.Tensor = th.stack(self.permutation, dim=-1)
        # self.inv_permutation: th.Tensor = th.stack(self.inv_permutation, dim=-1)
        self.permutation = nn.Parameter(th.stack(permutation, dim=-1), requires_grad=False)
        self.inv_permutation = nn.Parameter(th.stack(inv_permutation, dim=-1), requires_grad=False)

    def apply_permutation(self, x: th.Tensor) -> th.Tensor:
        """
        Randomize the input at each repetition according to `self.permutation`.

        Args:
            x: th.Tensor, last three dims must have shape [F, *, 1 or self.config.R]. * = any size

        Returns:
            th.Tensor: Randomized input along feature axis. Each repetition has its own permutation.
        """
        assert x.size(-1) == 1 or x.size(-1) == self.config.R
        if x.size(-1) == 1:
            x = x.repeat(*([1] * (x.dim()-1)), self.config.R)
        perm_indices = self.permutation.unsqueeze(-2).expand_as(x)
        x = th.gather(x, dim=-3, index=perm_indices)
        return x

    def apply_inv_permutation(self, x: th.Tensor) -> th.Tensor:
        """
        Apply the reverse permutation to the input.

        Args:
            x: th.Tensor, last three dims must have shape [F, *, 1 or self.config.R]. * = any size

        Returns:
            th.Tensor: Input with inverted permutation along feature axis. Each repetition has its own permutation.
        """
        assert x.size(-1) == 1 or x.size(-1) == self.config.R
        if x.size(-1) == 1:
            x = x.repeat(*([1] * (x.dim()-1)), self.config.R)
        perm_indices = self.inv_permutation.unsqueeze(-2).expand_as(x)
        x = th.gather(x, dim=-3, index=perm_indices)
        return x

    def forward(self, x: th.Tensor, layer_index: int = None, x_needs_permutation: bool = True,
                detach_params: bool = False) -> th.Tensor:
        """
        Forward pass through RatSpn.

        Args:
            x:
                Input of shape [*batch_dims, conditionals, self.config.F, output_channels, repetitions].
                    batch_dims: Sample shape per conditional.
                    conditionals: In CSPNs, weights are different for each conditional. In RatSpn, this is 1.
                    output_channels: self.config.I or 1 if x should be evaluated on all distributions of a leaf scope
                    repetitions: self.config.R or 1 if x should be evaluated on all repetitions
            layer_index: Evaluate log-likelihood of x at layer
            x_needs_permutation: An SPNs own samples where no inverted permutation was applied, don't need to be
                permuted in the forward pass.
            detach_params: If True, all SPN parameters involved in the forward pass are detached.
        Returns:
            th.Tensor: Log-likelihood of the input.
        """
        if layer_index is None:
            layer_index = self.max_layer_index

        assert x.dtype == self.dtype, \
            f"x has data type {x.dtype} but must have data type {self.dtype}"
        assert x.dim() >= 5, "Input needs at least 5 dims. Did you read the docstring of RatSpn.forward()?"
        assert x.size(-3) == self.config.F and (x.size(-2) == 1 or x.size(-2) == self.config.I) \
            and (x.size(-1) == 1 or x.size(-1) == self.config.R), \
            f"Shape of last three dims is {tuple(x.shape[3:])} but must be ({self.config.F}, 1 or {self.config.I}, " \
            f"1 or {self.config.R})"

        if x_needs_permutation:
            # Apply feature randomization for each repetition
            x = self.apply_permutation(x)

        # Apply leaf distributions
        x = self._leaf(x, detach_params=detach_params)

        # Forward to inner product and sum layers
        for layer in self._inner_layers[:layer_index]:
            x = layer(x, detach_params=detach_params)

        if layer_index == self.max_layer_index:
            # Merge results from the different repetitions into the channel dimension
            x = th.einsum('...dir -> ...dri', x)
            x = x.flatten(-2, -1).unsqueeze(-1)

            # Apply C sum node outputs
            x = self.root(x, detach_params=detach_params)

            # Remove repetition dimension
            # x = x.squeeze(-1)

            # Remove in_features dimension
            # x = x.squeeze(-3)

        return x

    def _build(self):
        """Construct the internal architecture of the RatSpn."""
        # Build the SPN bottom up:
        # Definition from RAT Paper
        # Leaf Region:      Create I leaf nodes
        # Root Region:      Create C sum nodes
        # Internal Region:  Create S sum nodes
        # Partition:        Cross products of all child-regions

        # Construct leaf
        self._leaf = self._build_input_distribution(gmm_leaves=self.config.gmm_leaves)

        self._inner_layers = nn.ModuleList()
        prod_in_channels = self.config.I

        # First product layer on top of leaf layer.
        # May pad output features of leaf layer is their number is not a power of 2.
        prodlayer = CrossProduct(
            in_features=self._leaf.out_features, in_channels=prod_in_channels, num_repetitions=self.config.R
        )
        self._inner_layers.append(prodlayer)
        sum_in_channels = self.config.I ** 2

        if self.config.D is None:
            self.config.D = int(np.log2(self.config.F))
        # Sum and product layers
        for i in np.arange(start=self.config.D - 1, stop=0, step=-1):
            # Current in_features
            in_features = 2 ** i

            # Sum layer
            sumlayer = Sum(in_features=in_features, in_channels=sum_in_channels, num_repetitions=self.config.R,
                           out_channels=self.config.S, dropout=self.config.dropout, ratspn=self.config.is_ratspn)
            self._inner_layers.append(sumlayer)

            # Product layer
            prodlayer = CrossProduct(in_features=in_features, in_channels=self.config.S, num_repetitions=self.config.R)
            self._inner_layers.append(prodlayer)

            # Update sum_in_channels
            sum_in_channels = self.config.S ** 2

        # Construct root layer
        self.root = Sum(
            in_channels=self.config.R * sum_in_channels, in_features=1, num_repetitions=1, out_channels=self.config.C,
            ratspn=self.config.is_ratspn,
        )

        # Construct sampling root with weights according to priors for sampling
        self._sampling_root = Sum(
            in_channels=self.config.C, in_features=1, out_channels=1, num_repetitions=1, ratspn=self.config.is_ratspn,
        )
        self._sampling_root.weight_param = nn.Parameter(
            th.ones(size=(1, 1, self.config.C, 1, 1)) * th.tensor(1 / self.config.C), requires_grad=False
        )

    def _build_input_distribution(self, gmm_leaves):
        """Construct the input distribution layer."""
        # Cardinality is the size of the region in the last partitions
        cardinality = np.ceil(self.config.F / (2 ** self.config.D)).astype(int)
        if gmm_leaves:
            return GaussianMixture(in_features=self.config.F, out_channels=self.config.I, gmm_modes=self.config.S,
                                   num_repetitions=self.config.R, cardinality=cardinality, dropout=self.config.dropout,
                                   tanh_squash=self.config.tanh_squash,
                                   leaf_base_class=self.config.leaf_base_class,
                                   leaf_base_kwargs=self.config.leaf_base_kwargs)
        else:
            return IndependentMultivariate(
                in_features=self.config.F, out_channels=self.config.I,
                num_repetitions=self.config.R, cardinality=cardinality, dropout=self.config.dropout,
                tanh_squash=self.config.tanh_squash,
                leaf_base_class=self.config.leaf_base_class,
                leaf_base_kwargs=self.config.leaf_base_kwargs,
                ratspn=self.config.is_ratspn,
            )

    @property
    def device(self):
        """Small hack to obtain the current device."""
        return self._sampling_root.weights.device

    def _init_weights(self):
        """Initiale the weights. Calls `_init_weights` on all modules that have this method."""
        for module in self.modules():
            if hasattr(module, "_init_weights") and module != self:
                module._init_weights()
                continue

            if isinstance(module, Sum):
                truncated_normal_(module.weights, std=0.5)
                continue

    @property
    def root_weights_split_by_rep(self):
        weights = self.root.weights
        w, d, _, oc, _ = weights.shape
        return weights.view(w, d, self.config.R, self.root.in_channels // self.config.R, oc).permute(0, 1, 3, 4, 2)

    def mpe(self, evidence: th.Tensor = None, layer_index=None) -> th.Tensor:
        """
        Perform MPE given some evidence.

        Args:
            evidence: Input evidence. Must contain some NaN values.
        Returns:
            th.Tensor: Clone of input tensor with NaNs replaced by MPE estimates.
        """
        return self.sample(mode='index', evidence=evidence, layer_index=layer_index, is_mpe=True).sample

    def sample(
            self, mode: str = None, n: Union[int, Tuple] = 1, class_index=None, evidence: th.Tensor = None,
            is_mpe: bool = False, layer_index: int = None, do_sample_postprocessing: bool = True,
            post_processing_kwargs: dict = None,
    ) -> Sample:
        """
        Sample from the distribution represented by this SPN.

        Args:
            mode: Two sampling modes are supported:
                'index': Sampling mechanism with indexes, which are non-differentiable.
                'onehot': This sampling mechanism work with one-hot vectors, grouped into tensors.
                          This way of sampling is differentiable, but also takes almost twice as long.
            n: Number of samples to generate.
            class_index: Class index. Can be either an int in combination with a value for `n` which will result in `n`
                samples from P(X | C = class_index). Or can be a list of ints which will map each index `c_i` in the
                list to a sample from P(X | C = c_i).
            evidence: th.Tensor with shape [optional batch_dims, w, F, r], where are is 1 if no repetition is specified.
                Evidence that can be provided to condition the samples.
                Evidence must contain NaN values which will be imputed according to the
                distribution represented by the SPN. The result will contain the evidence and replace all NaNs with the
                sampled values.
                If evidence is given, n is the number of samples to generate per evidence.
            is_mpe: Flag to perform max sampling (MPE).
            layer_index: Layer to start sampling from. None or self.max_layer_index = Root layer,
                self.num_layers-1 = Child layer of root layer, ...
            do_sample_postprocessing: If True, samples will be given to sample_postprocessing to fill in evidence,
                split by scope, be squashed, etc.
            post_processing_kwargs: Keyword args to the postprocessing.
                split_by_scope (bool), invert_permutation (bool)

        Returns:
            Sample with
                sample (th.Tensor): Samples generated according to the distribution specified by the SPN.
                    When sampling the root sum node of the SPN, the tensor
                    will be of size [nr_nodes, *batch_dims, w, f]:
                        nr_nodes: Dimension over the nodes being sampled. When sampling from the root sum node,
                            nr_nodes will be self.config.C (usually 1). When sampling from an inner layer,
                            the size of this dimension will be the number of that layer's output channels.
                        batch_dims: Shape with numbers of samples being drawn per conditional.
                            When sampling with evidence, batch_dims will be (*n, *evidence_batch_dims)
                        w: Dimension over conditionals.
                            In the RatSpn case, the class index and the evidence will set this dim. Otherwise,
                                it is always 1.
                            In the Cspn case, the SPN represents a conditional distribution.
                                w is the number of conditionals the Spn was given
                                prior to calling this function (see Cspn.sample()).
                        f: f = self.config.F is the number of features of the Spn as a whole.
                    Inner layers of the Spn have an additional repetition dimension. When sampling an inner layer, the
                    sample tensor will be of size [nr_nodes, n, w, f, r]
                        r: Dimension over the repetitions.
                    Sampling with split_by_scope = True adds another dimension [nr_nodes, s, n, w, f, r]
                        s: Number of scopes in the layer that is sampled from.
        """
        if post_processing_kwargs is None:
            post_processing_kwargs = {}
        assert mode is not None, "A sampling mode must be provided!"
        # assert class_index is None or evidence is None, "Cannot provide both, evidence and class indices."
        assert class_index is None or class_index.dim() == 1, "Class index must have shape [conditionals]"
        if class_index is not None and evidence is not None:
            assert class_index.shape[0] == evidence.shape[1], \
                "class_index must have same number of conditionals as evidence"
        assert evidence is None or evidence.dtype == self.dtype, \
            f"evidence has data type {evidence.dtype} but must have data type {self.dtype}"
        # assert n is None or evidence is None, "Cannot provide both, number of samples to generate (n) and evidence."

        if layer_index is None:
            layer_index = self.max_layer_index

        if isinstance(n, int):
            n = (n,)

        # Check if evidence contains nans
        if evidence is not None:
            assert (evidence != evidence).any(), "Evidence has no NaN values."
            # evidence needs shape
            # [*batch_dims, conditionals, self.config.F, output_channels, repetitions]
            # see RatSpn.forward() for more information
            n = (*n, *evidence.shape[:-4])

        with provide_evidence(self, evidence, requires_grad=(mode == 'onehot')):  # May be None but that's ok
            # If class is given, use it as base index
            # if class_index is not None:
                # Create new sampling context
                # raise NotImplementedError("This case has not been touched yet. Please verify.")
                # ctx = Sample(n=n,
                                      # parent_indices=class_index.repeat(n, 1).unsqueeze(-1).to(self.device),
                                      # repetition_indices=th.zeros((n, class_index.shape[0]), dtype=int, device=self.device),
                                      # is_mpe=is_mpe)
            ctx = Sample(
                n=n, is_mpe=is_mpe, sampling_mode=mode, needs_squashing=self.config.tanh_squash,
                sampled_with_evidence=evidence is not None,
            )

            if layer_index == self.max_layer_index:
                layer_index -= 1
                ctx.scopes = 1
                if class_index is not None:
                    ctx.parent_indices = th.as_tensor(class_index).expand(*n, -1)
                    if mode == 'onehot':
                        ctx.parent_indices = F.one_hot(ctx.parent_indices, num_classes=self.config.C)
                    ctx.parent_indices = ctx.parent_indices.unsqueeze(0).unsqueeze(-1)
                    ctx.parent_indices = ctx.parent_indices.unsqueeze(-2 if mode == 'index' else -3)

                ctx = self.root.sample(ctx=ctx, mode=mode)

                # mode == 'index'
                # ctx.parent_indices [nr_nodes, *batch_dims, w, d, r=1] contains indexes of
                # output channels of the next layer.
                # Sample from RatSpn root layer: Results are indices into the
                # stacked output channels of all repetitions

                # mode == 'onehot'
                # ctx.parent_indices [nr_nodes, *batch_dims, w, d, ic, r=1] contains indexes of
                # output channels of the next layer.
                # Sample from RatSpn root layer: Results are one-hot vectors of the indices
                # into the stacked output channels of all repetitions

                if mode == 'index':
                    ic_per_rep = self.root.weights.size(2) // self.config.R
                    ctx.repetition_indices = th.div(ctx.parent_indices, ic_per_rep, rounding_mode='trunc')
                    ctx.repetition_indices = ctx.repetition_indices.squeeze(-1).squeeze(-1)
                    # repetition_indices [nr_nodes, *sample_shape, w]
                    ctx.parent_indices = ctx.parent_indices % ic_per_rep
                    # parent_indices [nr_nodes, *sample_shape, w, d, r = 1]
                else:
                    ctx.parent_indices = ctx.parent_indices.view(*ctx.parent_indices.shape[:-2], self.config.R, -1)
                    ctx.parent_indices = th.einsum('...ri -> ...ir', ctx.parent_indices)
                    # ctx.parent_indices [nr_nodes, *sample_shape, w, d, ic, r]
                ctx.has_rep_dim = False  # The final sample will be of one repetition only
            ctx.assert_correct_indices()

            # Continue at layers
            # Sample inner layers in reverse order (starting from topmost)
            # noinspection PyTypeChecker
            for layer in reversed(self._inner_layers[:layer_index]):
                if ctx.scopes is None:
                    if isinstance(layer, Sum):
                        ctx.scopes = layer.in_features
                    else:
                        ctx.scopes = layer.in_features // layer.cardinality
                ctx = layer.sample(ctx=ctx, mode=mode)
                try:
                    ctx.assert_correct_indices()
                except AssertionError as e:
                    raise AssertionError(f"In layer {layer}, sampling mode {ctx.sampling_mode}:\n{e}")

            # Sample leaf
            if ctx.scopes is None:
                ctx.scopes = self._leaf.in_features // self._leaf.cardinality
            samples = self._leaf.sample(ctx=ctx, mode=mode)
            if layer_index == 0:
                samples = th.einsum('nwFIR -> InwFR', samples)
            ctx.sample = samples

        if do_sample_postprocessing:
            ctx = self.sample_postprocessing(ctx, evidence, **post_processing_kwargs)

        return ctx

    def sample_postprocessing(
            self, ctx: Sample, evidence: th.Tensor = None, split_by_scope: bool = False,
            invert_permutation: bool = True,
    ) -> Sample:
        """
        Apply postprocessing to samples.

        Args:
            ctx: Sample returned by sample().
            evidence: Evidence that can be provided to condition the samples. If evidence is given, `n` and
                `class_index` must be `None`. Evidence must contain NaN values which will be imputed according to the
                distribution represented by the SPN. The result will contain the evidence and replace all NaNs with the
                sampled values.
            split_by_scope: Arg to postprocessing: In an inner layer, a node's sample doesn't
                cover the entire feature set f, but rather a scope of it. split_by_scope allows makes the
                scope of a single node's samples visible in the entire Spn's scope, as features in f not
                covered by this node are filled with NaNs.
            invert_permutation: Invert permutation of data features.

        Returns:
            A modified Sample. See doc string of sample() for shape information.
        """
        sample = ctx.sample
        if split_by_scope:
            assert not ctx.is_split_by_scope, "Samples are already split by scope!"
            assert self.config.F % ctx.scopes == 0, "Size of entire scope is not divisible by the number of" \
                                                    " scopes in the layer we are sampling from. What do?"
            features_per_scope = self.config.F // ctx.scopes

            sample = sample.unsqueeze(1)
            sample = sample.repeat(1, ctx.scopes, *([1] * (sample.dim()-2)))

            for i in range(ctx.scopes):
                mask = th.ones(self.config.F, dtype=th.bool)
                mask[i * features_per_scope:(i + 1) * features_per_scope] = False
                sample[:, i, :, :, mask, ...] = th.nan
            ctx.is_split_by_scope = True

        # Each repetition has its own inverted permutation which we now apply to the samples.
        if invert_permutation:
            assert not ctx.permutation_inverted, "Permutations are already inverted on the samples!"
            if ctx.sampling_mode == 'index':
                if ctx.repetition_indices is not None:
                    rep_sel_inv_perm = self.inv_permutation.T[ctx.repetition_indices]
                else:
                    rep_sel_inv_perm = self.inv_permutation.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    # The squeeze(1) is only for the case that split_by_scope is True and ctx.scopes == 1
                    # rep_sel_inv_perm = rep_sel_inv_perm.expand(*sample.shape[:-2], -1, -1).squeeze(1)
            else:
                if ctx.parent_indices is not None:
                    rep_sel_inv_perm = self.inv_permutation * ctx.parent_indices.detach().sum(-2).long()
                    if not ctx.has_rep_dim:
                        rep_sel_inv_perm = rep_sel_inv_perm.sum(-1)
                else:
                    rep_sel_inv_perm = self.inv_permutation.unsqueeze(0).unsqueeze(0).unsqueeze(0)

            if ctx.is_split_by_scope:
                rep_sel_inv_perm = rep_sel_inv_perm.unsqueeze(1)
            rep_sel_inv_perm = rep_sel_inv_perm.expand_as(sample)
            sample = th.gather(sample, dim=-2 if ctx.has_rep_dim else -1, index=rep_sel_inv_perm)
            ctx.permutation_inverted = True

            if not ctx.has_rep_dim:
                sample = sample.unsqueeze(-1)
                ctx.has_rep_dim = True

        if evidence is not None:
            # evidence has shape [*evidence_batch_dims, w, d, oc, r],
            # sample has shape [oc, *sample_batch_dims, *evidence_batch_dims, w, d, r]
            evidence = evidence.expand(*ctx.n, -1, -1, -1, -1)
            evidence = th.einsum('...or -> o...r', evidence)
            assert evidence.shape == sample.shape

            # Update NaN entries in evidence with the sampled values
            fill_with_evidence = ~th.isnan(evidence)
            sample[fill_with_evidence] = evidence[fill_with_evidence]
            ctx.evidence_filled_in = True

        if self.config.tanh_squash and ctx.needs_squashing:
            sample = sample.clamp(-6.0, 6.0).tanh()
            ctx.needs_squashing = False

        ctx.sample = sample
        return ctx

    def sample_index_style(self, **kwargs):
        return self.sample(mode='index', **kwargs)

    def sample_onehot_style(self, **kwargs):
        return self.sample(mode='onehot', **kwargs)

    def weigh_tensors(self, layer_index, tensors: List = None, return_weight_ent=False) \
            -> Tuple[List[th.Tensor], Optional[th.Tensor]]:
        """
        Weigh each tensor in a list with the weights of a Sum layer.
        This function takes care of the tricky case of the root sum node.

        Args:
            layer_index: Index of layer to take weights from. Must be a Sum layer.
            tensors: A list of tensors to multiply the weights with.
            return_weight_ent: If True, the weight entropy also computed.

        Returns:
            tensors: List of the weighted input tensors.
            weight_entropy: Either the calculated weight entropy or None.
        """
        layer = self.layer_index_to_obj(layer_index)
        assert isinstance(layer, Sum), "Responsibilities can only be computed for Sum layers!"

        layer_log_weights = layer.weights
        weights = layer_log_weights.exp()
        weight_entropy = th.sum(-weights * layer_log_weights, dim=2) if return_weight_ent else None

        if layer_index < self.max_layer_index:
            tensors = [th.sum(weights * t, dim=2) for t in tensors]
        else:
            w, _, ic, oc, _ = weights.shape
            r = self.config.R
            root_weights_over_rep = weights.view(w, 1, r, ic // r, oc)
            weights = th.einsum('wdrio -> wdior', root_weights_over_rep)
            tensors = [(weights * t).sum(dim=-3).sum(dim=-1) for t in tensors]
            weight_entropy = weight_entropy.squeeze(-1)

        return tensors, weight_entropy

    def layer_entropy_approx(
            self, layer_index=0, child_entropies: th.Tensor = None, child_ll: th.Tensor = None,
            sample_size=5, aux_with_grad=False, verbose=False, marginal_mask=None,
    ) -> Tuple[th.Tensor, Optional[Dict]]:
        """

        Args:
            layer_index:
            child_entropies: th.Tensor of shape [w, d, oc of child layer == ic of this layer, r]
            child_ll: If provided, children don't need to be sampled. Shape [n, w, d, i, r]
            sample_size:
            aux_with_grad: If False, the approximation of the responsibility is done without grad.
            verbose: If True, return dict of entropy metrics

        Returns: Tuple of
            node_entropies: th.Tensor of size [w, d, oc of child layer == ic of this layer, r]
            logging: Dict or None, depending on verbose
        """
        assert child_entropies is not None or layer_index == 0, \
            "When sampling from a layer other than the leaf layer, child entropies must be provided!"
        logging = {}
        metrics = {}
        if marginal_mask is not None:
            assert (shape_is := marginal_mask.shape) == (shape_should := self._leaf.base_leaf.mean_param.shape[:2]), \
                f"marginal_mask is of shape {shape_is} but needs to have shape {shape_should}"
            marginal_mask = marginal_mask.clone().bool().unsqueeze(-1).unsqueeze(-1)
            marginal_mask = self.apply_permutation(marginal_mask)

        if layer_index == 0:
            if child_ll is None:
                ctx = self.sample(
                    mode='onehot' if th.is_grad_enabled() else 'index',
                    n=sample_size, layer_index=layer_index, is_mpe=False,
                    do_sample_postprocessing=False,
                )
                sample = th.einsum('InwFR -> nwFIR', ctx.sample)
                if marginal_mask is not None:
                    sample[marginal_mask.expand_as(sample)] = th.nan
                # ctx.sample [self.config.I, n, w, self.config.F, self.config.R]
                child_ll = self.forward(
                    x=sample, layer_index=layer_index, x_needs_permutation=False
                )
            # child_ll [n, w, self.config.F // leaf cardinality, self.config.I, self.config.R]
            node_entropies = -child_ll.mean(dim=0)
        else:
            layer = self.layer_index_to_obj(layer_index)

            if isinstance(layer, CrossProduct):
                node_entropies = layer(child_entropies)
            else:
                if child_ll is None:
                    ctx = self.sample(
                        mode='onehot' if th.is_grad_enabled() else 'index',
                        n=sample_size, layer_index=layer_index - 1, is_mpe=False,
                        do_sample_postprocessing=False,
                    )
                    samples = ctx.sample
                    if samples.dim() == 4:
                        samples = samples.unsqueeze(-1)
                    samples = samples.unsqueeze(-2)
                    # We sampled all ic nodes of each repetition in the child layer
                    if marginal_mask is not None:
                        samples[marginal_mask.expand_as(samples)] = th.nan

                    child_ll = self.forward(x=samples, layer_index=layer_index - 1, x_needs_permutation=False,
                                            detach_params=not aux_with_grad)
                    child_ll = child_ll.mean(dim=1)

                ic, w, d, _, r = child_ll.shape
                if layer_index == self.max_layer_index:  # layer is root sum layer
                    # Now we are dealing with a log-likelihood tensor with the shape [ic, w, 1, ic, r],
                    # where child_ll[0,0,:,:,0] are the log-likelihoods of the ic nodes in the first repetition
                    # given the samples from the first node of that repetition.
                    # The problem is that the weights of the root sum node don't recognize different, separated
                    # repetitions, so we reshape the weights to make the repetition dimension explicit again.
                    # This is equivalent to splitting up the root sum node into one sum node per repetition,
                    # with another sum node sitting on top.
                    log_weights = self.root_weights_split_by_rep if aux_with_grad else self.root_weights_split_by_rep.detach()
                    # log_weights = th.log_softmax(root_weights_over_rep, dim=2)
                    ll = child_ll.unsqueeze(-2) + log_weights
                    ll = th.logsumexp(ll, dim=-3)
                else:
                    ll = layer.forward(child_ll, detach_params=not aux_with_grad)
                    log_weights = layer.weights if aux_with_grad else layer.weights.detach()

                    # We have the log-likelihood of the current sum layer w.r.t. the samples from its children.
                ll = th.einsum('iwdoR -> wdioR', ll)

                # child_ll now contains the log-likelihood of the samples from all of its 'ic' nodes per feature and
                # repetition - ic * d * r in total.
                # child_ll contains the LL of the samples of each node evaluated among all other nodes - separated
                # by repetition and feature.
                # The tensor shape is [ic, w, d, ic, r]. Looking at one conditional, one feature and one repetition,
                # we are looking at the slice [:, 0, 0, :, 0].
                # The first dimension is the dimension of the samples - there are 'ic' of them.
                # The 4th dimension is the dimension of the LLs of the nodes for those samples.
                # So [4, 0, 0, :, 0] contains the LLs of all nodes given the sample from the fifth node.
                # Likewise, [:, 0, 0, 2, 0] contains the LLs of the samples of all nodes, evaluated at the third node.
                # We needed the full child_ll tensor to compute the LLs of the current layer, but now we only
                # require the LLs of each node's own samples.
                child_ll = child_ll.unsqueeze(-2)
                child_ll = th.einsum('iwdioR -> wdioR', child_ll)

                responsibility: th.Tensor = log_weights + child_ll - ll
                # responsibility [w, d, ic, oc, r]

                [weighted_ch_ents, weighted_aux_resp], weight_entropy = self.weigh_tensors(
                    layer_index=layer_index,
                    tensors=[child_entropies.unsqueeze(3), responsibility],
                    return_weight_ent=True
                )
                node_entropies = weight_entropy + weighted_ch_ents + weighted_aux_resp

                if verbose:
                    metrics.update({
                        'weight_entropy': weight_entropy.detach(),
                        'responsibility': responsibility.detach(),
                    })

        if verbose:
            metrics.update({
                'node_entropy': node_entropies.detach()
            })
            logging = self.log_dict_from_metric(layer_index, metrics)

        return node_entropies, logging

    def log_dict_from_metric(self, layer_index: int, metrics: Dict, rep_dim=-1, batch_dim: Optional[int] = 0):
        log_dict = {}
        for rep in range(list(metrics.values())[0].size(rep_dim)):
            rep_key = f"rep{rep}"
            rep = th.as_tensor(rep, device=self.device)
            for key, metric in metrics.items():
                if metric is None:
                    continue
                key_base = f"lay{layer_index}/{rep_key}/{key}"
                metric = metric.to(self.device)
                log_dict.update({
                    f"{key_base}/min": metric.index_select(rep_dim, rep).min().item(),
                    f"{key_base}/max": metric.index_select(rep_dim, rep).max().item(),
                    f"{key_base}/mean": metric.index_select(rep_dim, rep).mean().item(),
                })
                if batch_dim is not None:
                    std = metric.index_select(rep_dim, rep).std(dim=batch_dim).mean().item()
                    if not np.isnan(std):
                        log_dict.update({
                            f"{key_base}/std": std
                        })

        return log_dict

    def recursive_entropy_approx(
            self, sample_size=10, aux_with_grad: bool = False, verbose=False, marginal_mask: th.Tensor = None,
    ) -> Tuple[th.Tensor, Optional[Dict]]:
        """
        Approximate the entropy of the root sum node in a recursive manner.

        Args:
            sample_size: Number of samples to approximate the expected entropy of the responsibility with.
            aux_with_grad: If False, the approximation of the responsibility is done without grad.
            verbose: Return logging data
        """
        logging = {}
        child_entropies = None
        for layer_index in range(self.num_layers):
            child_entropies, layer_log = self.layer_entropy_approx(
                layer_index=layer_index, child_entropies=child_entropies,
                sample_size=sample_size, aux_with_grad=aux_with_grad,
                verbose=verbose, marginal_mask=marginal_mask,
            )
            if layer_log != {}:
                logging.update(layer_log)
        return child_entropies.flatten(), logging

    def naive_entropy_approx(
            self, sample_size=100, layer_index: int = None, sample_with_grad=False, marginal_mask: th.Tensor = None,
    ):
        if layer_index is None:
            layer_index = self.max_layer_index
        if marginal_mask is not None:
            assert (shape_is := marginal_mask.shape) == (shape_should := self._leaf.base_leaf.mean_param.shape[:2]), \
                f"marginal_mask is of shape {shape_is} but needs to have shape {shape_should}"
            marginal_mask = marginal_mask.clone().bool()

        sample_args = {
            'n': sample_size, 'layer_index': layer_index, 'do_sample_postprocessing': False
        }
        if sample_with_grad:
            samples = self.sample(mode='onehot', **sample_args)
        else:
            with th.no_grad():
                samples = self.sample(mode='index', **sample_args)

        samples = samples.sample
        if marginal_mask is not None:
            samples[marginal_mask.expand_as(samples)] = th.nan
        samples = th.einsum('o...d -> ...do', samples).unsqueeze(-1)
        log_probs = self.forward(samples, layer_index=layer_index)
        return -log_probs.mean()

    def huber_entropy_lb(self, layer_index: int = None, verbose=True,
                         detach_weights: bool = False, add_sub_weight_ent: bool = False,
                         detach_weight_ent_subtraction: bool = False, marginal_mask: th.Tensor = None):
        """
        Calculate the entropy lower bound of the SPN as in Huber '08.

        Args:
            layer_index: Compute the entropy LB for this layer. Can take values from 1 to self.max_layer_index+1.
                self.max_layer_index+1 is the sampling root.
            detach_weights: If True, the sum weights are detached for computing the entropy lower bound.
            add_sub_weight_ent: If True, the weight entropies are added and subtracted to the lower bound to
                introduce a grad that keeps weight entropies of intermediate layers up
            detach_weight_ent_subtraction: If True, the subtraction of the weight_entropy is detached instead of the
                addition.
        Returns:
            Tuple of entropy lower bound and logging dict

        """
        logging = {}
        entropy_lb = None
        if layer_index is None:
            layer_index = self.max_layer_index
        if marginal_mask is not None:
            assert (shape_is := marginal_mask.shape) == (shape_should := self._leaf.base_leaf.mean_param.shape[:2]), \
                f"marginal_mask is of shape {shape_is} but needs to have shape {shape_should}"
            marginal_mask = marginal_mask.clone().bool()

        # Distribution layer of leaf layer
        means = self.apply_inv_permutation(self.means)
        var = self.apply_inv_permutation(self.var)
        var = var.unsqueeze(2).unsqueeze(-2) + var.unsqueeze(3).unsqueeze(-1)
        std = th.sqrt(var)
        gauss = dist.Normal(means.unsqueeze(2).unsqueeze(-2).expand_as(std), std)
        log_probs = gauss.log_prob(means.unsqueeze(3).unsqueeze(-1).expand_as(std))
        if marginal_mask is not None:
            marginal_mask = marginal_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            log_probs[marginal_mask.expand_as(log_probs)] = 0.0
        permutation = self.permutation.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2).expand_as(log_probs)
        log_probs = th.gather(log_probs, dim=1, index=permutation)

        # Product layer of leaf layer
        w, F, o1, o2, r1, r2 = log_probs.shape
        d_out = F // self._leaf.prod.cardinality
        log_probs = log_probs.view(w, d_out, self._leaf.prod.cardinality, o1, o2, r1, r2)
        log_probs = log_probs.sum(2)

        for i in range(1, layer_index+1):
            if i % 2 == 1 and i != self.max_layer_index+1:
                # It is a CrossProduct layer
                # Calculate the log_probs of the product nodes among themselves
                left_scope, right_scope = CrossProduct.split_shuffled_scopes(log_probs, 1)
                w, d, o, _, R, _ = left_scope.shape  # first dim is o as well
                left_scope = left_scope.unsqueeze(2).unsqueeze(-4)
                right_scope = right_scope.unsqueeze(3).unsqueeze(-3)
                if i == self.max_layer_index-1 and False:
                    left_scope = left_scope.unsqueeze(-2)
                    right_scope = right_scope.unsqueeze(-1)
                    log_probs = left_scope + right_scope
                    log_probs = th.einsum('ab...cdij -> jab...icd', log_probs)
                    log_probs = log_probs.reshape(o**2 * R, w, d, o**2 * R, 1)
                else:
                    log_probs = left_scope + right_scope
                    log_probs = log_probs.view(w, d, o**2, o**2, R, R)
            else:
                # Add log weights to log probs and sum in linear space
                layer = self.layer_index_to_obj(i)
                weight_entropy = - th.sum(layer.weights * layer.weights.exp(), dim=2)
                if i < self.max_layer_index:
                    weights = layer.weights
                elif i == self.max_layer_index:
                    weights = self.root_weights_split_by_rep
                else:
                    weights = self._sampling_root.weights
                if detach_weights or i > self.max_layer_index:
                    weights = weights.detach()

                # weights = weights.unsqueeze(2).unsqueeze(-3).unsqueeze(-2) + weights.unsqueeze(3).unsqueeze(-2).unsqueeze(-1)
                # log_probs = log_probs.unsqueeze(-3).unsqueeze(-3) + weights
                # log_probs = log_probs.logsumexp(2).logsumexp(2)
                log_probs = log_probs.unsqueeze(-3) + weights.unsqueeze(2).unsqueeze(-2)
                # We don't logsumexp over the unsqueezed dims
                log_probs = log_probs.logsumexp(3)

                if i <= self.max_layer_index and add_sub_weight_ent:
                    we_term = weight_entropy.unsqueeze(2).unsqueeze(-1)
                    if detach_weight_ent_subtraction:
                        log_probs = log_probs - we_term.detach() + we_term
                    else:
                        log_probs = log_probs - we_term + we_term.detach()

                if i == self.max_layer_index:
                    log_probs = log_probs.logsumexp(-1)

                if verbose or i == layer_index:
                    if i < self.max_layer_index:
                        probs_for_ent_lb = th.einsum('...ii -> ...i', log_probs)
                    else:
                        probs_for_ent_lb = log_probs
                    entropy_lb = - th.sum(weights.exp() * probs_for_ent_lb, dim=-3)
                    if i == self.max_layer_index:
                        entropy_lb = entropy_lb.sum(-1)
                if i < layer_index:
                    log_probs = log_probs.unsqueeze(-3) + weights.unsqueeze(-3).unsqueeze(-1)
                    log_probs = log_probs.logsumexp(2)

                if verbose:
                    metrics = {
                        'weight_entropy': weight_entropy.detach(),
                        'huber_entropy_LB': entropy_lb.detach(),
                    }
                    logging.update(self.log_dict_from_metric(i, metrics, batch_dim=None))
                continue
                log_probs = log_probs.unsqueeze(-2) + weights.unsqueeze(0)
                log_probs = log_probs.logsumexp(-3)
                log_probs = th.einsum('iwdkR -> wdikR', log_probs)

                if verbose or i == layer_index:
                    entropy_lb = - th.sum(weights.exp() * log_probs, dim=-3)
                if i < layer_index:
                    log_probs = log_probs.unsqueeze(-2) + weights.unsqueeze(-3)
                    log_probs = log_probs.logsumexp(2)
                    log_probs = th.einsum('wdklR -> kwdlR', log_probs)

                if verbose:
                    weight_entropy = - th.sum(weights * weights.exp(), dim=2).unsqueeze(0)
                    metrics = {
                        'weight_entropy': weight_entropy.detach(),
                        'huber_entropy_LB': entropy_lb.detach(),
                    }
                    logging[i] = self.log_dict_from_metric(metrics, batch_dim=None)

        return entropy_lb, logging

    @property
    def is_ratspn(self):
        return self.config.is_ratspn

    @property
    def means(self):
        return self._leaf.base_leaf.means

    @means.setter
    def means(self, means: th.Tensor):
        self._leaf.base_leaf.means = means

    @property
    def stds(self):
        return self._leaf.base_leaf.stds

    @property
    def log_stds(self):
        return self._leaf.base_leaf.log_stds

    @log_stds.setter
    def log_stds(self, log_stds: th.Tensor):
        self._leaf.base_leaf.log_stds = log_stds

    @property
    def var(self):
        return self._leaf.base_leaf.var

    def debug__set_root_weights_dirac(self):
        self.debug__set_weights_dirac(self.max_layer_index)

    def debug__set_weights_dirac(self, layer_index):
        layer = self.layer_index_to_obj(layer_index)
        assert isinstance(layer, Sum), "Given layer index is not a Sum layer!"
        weights = layer.weights
        weights[:] = -100.0
        weights[0, 0, 0, :, 0] = -1e-3
        weights = weights.log_softmax(dim=2)
        if isinstance(layer.weight_param, nn.Parameter):
            weights = nn.Parameter(weights)
            del layer.weight_param
        layer.weight_param = weights

    def debug__set_weights_uniform(self, layer_index):
        layer = self.layer_index_to_obj(layer_index)
        assert isinstance(layer, Sum), "Given layer index is not a Sum layer!"
        weights = layer.weights
        weights[:] = -100.0
        weights = weights.log_softmax(dim=2)
        if isinstance(layer.weight_param, nn.Parameter):
            weights = nn.Parameter(weights)
            del layer.weight_param
        layer.weight_param = weights

    def debug__set_dist_params(self, min_mean = -100.0, max_mean = 100.0):
        """
            Set the dist parameters for debugging purposes.
            Std is set to a very low value.
        """
        means = th.arange(min_mean, max_mean, (max_mean - min_mean) / self._leaf.base_leaf.means.numel(), device=self.device)
        means = means.reshape_as(self._leaf.base_leaf.means)
        log_stds = self._leaf.base_leaf.log_stds * 0.0 - 20.0
        if isinstance(self._leaf.base_leaf.mean_param, nn.Parameter):
            means = nn.Parameter(means)
            log_stds = nn.Parameter(log_stds)
            del self._leaf.base_leaf.mean_param
            del self._leaf.base_leaf.log_std_param
        self._leaf.base_leaf.mean_param = means
        self._leaf.base_leaf.log_std_param = log_stds

    def shape_like_crossprod_input_mapping(self, t: th.Tensor, ic_dim=-3):
        """
            In a RAT-SPN, the cross-product layer takes two scopes, the "left scope" (:= ls)
            and the "right scope" (:= rs), both with I channels,
            and maps them in the following way to its I^2 output channels:
            [(ls ch. 0, rs ch. 0), (ls ch. 0, rs ch. 1), ..., (ls ch. 0, rs. ch I-1),
             (ls ch. 1, rs ch. 0), (ls ch. 1, rs ch. 1), ..., (ls ch. 1, rs. ch I-1),
             ...,
             (ls ch. I-1, rs ch. 0), (ls ch. I-1, rs ch. 1), ..., (ls ch. I-1, rs. ch I-1)]

             This function takes the I^2 weights of a sum layer and shapes into "left scope weights"
             and "right scope weights".
        """
        before_ic = t.shape[:ic_dim]
        after_ic = t.shape[(ic_dim+1):]
        ic = t.size(ic_dim)
        ic_new = np.sqrt(ic).astype(int)
        t = t.view(*before_ic, ic_new, ic_new, *after_ic)
        # In the ic dim, t[0,0,:,0,0] = [w_0, w_1, w_2, ..., w_{ic-1}] becomes t[0,0,:,:,0,0] =
        #    [[w_0, w_1, ..., w_{ic_new-1}],
        #     [w_{ic_new}, ...,w_{2*ic_new-1}], and so on with ic_new rows in total.
        return t

    def save(self, *args, **kwargs):
        th.save(self, *args, **kwargs)
