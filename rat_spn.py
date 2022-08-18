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
import more

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
    tanh_squash: bool = False

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
        self.D = check_valid(self.D, int, 1)
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
        self._init_weights()

        # Obtain permutation indices
        self._make_random_repetition_permutation_indices()

        self.__max_layer_index = len(self._inner_layers) + 1
        self.num_layers = self.__max_layer_index + 1

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

    def forward(self, x: th.Tensor, layer_index: int = None, x_needs_permutation: bool = True) -> th.Tensor:
        """
        Forward pass through RatSpn.

        Args:
            x:
                Input of shape [*batch_dims, weight_sets, self.config.F, output_channels,
                                self.config.R or 1 if no rep is to be specified].
                    batch_dims: Sample shape per weight set (= per conditional in the CSPN sense).
                    weight_sets: In CSPNs, weights are different for each conditional. In RatSpn, this is 1.
                    output_channels: self.config.I or 1 if x should be evaluated on each distribution of a leaf scope
            layer_index: Evaluate log-likelihood of x at layer
            x_needs_permutation: An SPNs own samples where no inverted permutation was applied, don't need to be
                permuted in the forward pass.
        Returns:
            th.Tensor: Log-likelihood of the input.
        """
        if layer_index is None:
            layer_index = self.max_layer_index

        assert x.dim() >= 5, "Input needs at least 5 dims. Did you read the docstring of RatSpn.forward()?"
        assert x.size(-3) == self.config.F and (x.size(-2) == 1 or x.size(-2) == self.config.I) \
            and (x.size(-1) == 1 or x.size(-1) == self.config.R), \
            f"Shape of last three dims is {tuple(x.shape[3:])} but must be ({self.config.F}, 1 or {self.config.I}, " \
            f"1 or {self.config.R})"

        if x_needs_permutation:
            # Apply feature randomization for each repetition
            x = self.apply_permutation(x)

        # Apply leaf distributions
        x = self._leaf(x)

        # Forward to inner product and sum layers
        for layer in self._inner_layers[:layer_index]:
            x = layer(x)

        if layer_index == self.max_layer_index:
            # Merge results from the different repetitions into the channel dimension
            x = th.einsum('...dir -> ...dri', x)
            x = x.flatten(-2, -1).unsqueeze(-1)

            # Apply C sum node outputs
            x = self.root(x)

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
            evidence: th.Tensor with shape [*batch_dims, w, F, r], where are is 1 if no repetition is specified.
                Evidence that can be provided to condition the samples. If evidence is given, `n` and
                `class_index` must be `None`. Evidence must contain NaN values which will be imputed according to the
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
                    When sampling the root sum node of the SPN, the tensor will be of size [nr_nodes, n, w, f]:
                        nr_nodes: Dimension over the nodes being sampled. When sampling from the root sum node,
                            nr_nodes will be self.config.C (usually 1). When sampling from an inner layer,
                            the size of this dimension will be the number of that layer's output channels.
                        n: Number of samples being drawn per weight set (dimension `w`).
                        w: Dimension over weight sets. In the RatSpn case, w is always 1. In the Cspn case, the SPN
                            represents a conditional distribution. w is the number of conditionals the Spn was given
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
        assert class_index is None or evidence is None, "Cannot provide both, evidence and class indices."
        # assert n is None or evidence is None, "Cannot provide both, number of samples to generate (n) and evidence."

        if layer_index is None:
            layer_index = self.max_layer_index

        if isinstance(n, int):
            n = (n,)

        # Check if evidence contains nans
        if evidence is not None:
            assert (evidence != evidence).any(), "Evidence has no NaN values."
            assert evidence.shape[-2] == self.config.F and \
                   (evidence.shape[-1] == self.config.R or evidence.shape[-1] == 1), \
                "The evidence doesn't seem to have a repetition dimension."
            assert evidence.shape[-3] == self.root.weights.shape[0], \
                "Evidence was created under a different number of conditionals than what the SPN is set to now. "
            # evidence has shape [*batch_dims, w, self.config.F, 1 or self.config.R]

            # Set n to the number of samples in the evidence
            n = (*n, *evidence.shape[:-3])

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
                    n = (*n, len(class_index))
                    ctx.parent_indices = th.as_tensor(class_index).expand(*n)
                    ctx.parent_indices = th.einsum('...c -> c...', ctx.parent_indices)
                    if mode == 'onehot':
                        ctx.parent_indices = F.one_hot(ctx.parent_indices, num_classes=self.config.C)
                    ctx.n = n
                    ctx.parent_indices = ctx.parent_indices.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

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

        if self.config.tanh_squash and ctx.needs_squashing:
            sample = sample.clamp(-6.0, 6.0).tanh()
            ctx.needs_squashing = False

        if evidence is not None:
            if self.config.tanh_squash:
                evidence = evidence.clamp(-6.0, 6.0).tanh()
            # Update NaN entries in evidence with the sampled values
            nan_mask = th.isnan(evidence).expand_as(sample)
            evidence = evidence.expand_as(sample).clone()

            # First make a copy such that the original object is not changed
            evidence[nan_mask] = sample[nan_mask.expand_as(sample)]
            sample = evidence
            ctx.evidence_filled_in = True

        ctx.sample = sample
        return ctx

    def sample_index_style(self, **kwargs):
        return self.sample(mode='index', **kwargs)

    def sample_onehot_style(self, **kwargs):
        return self.sample(mode='onehot', **kwargs)

    def resp_playground(self, layer_index, child_ll: th.Tensor) -> th.Tensor:
        """
        Approximate the responsibilities of a Sum layer in the Spn.
        For this it draws samples of the child nodes of the layer. These samples can be returned as well.

        Args:
            layer_index: The layer index to evaluate. The layer must be a Sum layer.
            child_ll: th.Tensor of child log-likelihoods.

        Returns: Tuple of
            responsibilities: th.Tensor of shape [w, d, oc of child layer, oc, r of child layer].
        """
        layer = self.layer_index_to_obj(layer_index)
        assert isinstance(layer, Sum), "Responsibilities can only be computed for Sum layers!"

        r = self.config.R
        ic = layer.in_channels // r
        w, d = layer.weight_param.shape[:2]
        assert child_ll.shape == (ic, w, d, ic, r)

        if layer_index == self.max_layer_index:  # layer is root sum layer
            # Now we are dealing with a log-likelihood tensor with the shape [ic, w, 1, ic, r],
            # where child_ll[0,0,:,:,0] are the log-likelihoods of the ic nodes in the first repetition
            # given the samples from the first node of that repetition.
            # The problem is that the weights of the root sum node don't recognize different, separated
            # repetitions, so we reshape the weights to make the repetition dimension explicit again.
            # This is equivalent to splitting up the root sum node into one sum node per repetition,
            # with another sum node sitting on top.
            root_weights_over_rep = layer.weights.view(w, 1, r, ic).permute(0, 1, 3, 2).unsqueeze(-2)
            log_weights = th.log_softmax(root_weights_over_rep, dim=2)
            # child_ll and the weights are log-scaled, so we add them together.
            ll = child_ll.unsqueeze(-2) + log_weights.unsqueeze(0)
            # ll shape [ic, w, 1, ic, r]
            ll = th.logsumexp(ll, dim=3)
        else:
            ll = layer(child_ll)
            log_weights = layer.weights

            # We have the log-likelihood of the current sum layer w.r.t. the samples from its children.
            # We permute the dims so this tensor is of shape [w, d, ic, oc, r]
        ll = ll.permute(1, 2, 0, 3, 4)

        # child_ll now contains the log-likelihood of the samples from all of its 'ic' nodes per feature and
        # repetition - ic * d * r in total.
        # child_ll contains the LL of the samples of each node evaluated among all other nodes - separated
        # by repetition and feature.
        # The tensor shape is [ic, w, d, ic, r]. Looking at one weight set, one feature and one repetition,
        # we are looking at the slice [:, 0, 0, :, 0].
        # The first dimension is the dimension of the samples - there are 'ic' of them.
        # The 4th dimension is the dimension of the LLs of the nodes for those samples.
        # So [4, 0, 0, :, 0] contains the LLs of all nodes given the sample from the fifth node.
        # Likewise, [:, 0, 0, 2, 0] contains the LLs of the samples of all nodes, evaluated at the third node.
        # We needed the full child_ll tensor to compute the LLs of the current layer, but now we only
        # require the LLs of each node's own samples.
        child_ll = child_ll[range(ic), :, :, range(ic), :]  # [ic, w, d, r]
        child_ll = child_ll.permute(1, 2, 0, 3)
        child_ll = child_ll.unsqueeze(3)  # [w, d, ic, 1, r]

        responsibilities: th.Tensor = log_weights + child_ll - ll
        return responsibilities

    def layer_responsibilities(self, layer_index, sample_size: int = 5, child_ll: th.Tensor = None,
                                return_sample_ctx: bool = False, with_grad=False) \
            -> Tuple[th.Tensor, Optional[Sample]]:
        """
        Approximate the responsibilities of a Sum layer in the Spn.
        For this it draws samples of the child nodes of the layer. These samples can be returned as well.

        Args:
            layer_index: The layer index to evaluate. The layer must be a Sum layer.
            sample_size: Number of samples to evaluate responsibilities on.
                Responsibilities can't be computed in closed form.
            child_ll: th.Tensor of child log-likelihoods. If child_ll is None, the children of the layer are sampled
                and the log-likelihoods are computed from these samples. If child_ll is given, return_sample_ctx and
                with_grad must be False. Shape [ic, w, d, ic, r] (mean already taken over sample dim).
            return_sample_ctx: If True, the sample context including the samples are returned. No post-processing is
                applied to these.
            with_grad: If True, sampling is done in a differentiable way.

        Returns: Tuple of
            responsibilities: th.Tensor of shape [w, d, oc of child layer, oc, r of child layer].
            samples: th.Tensor of shape [ic, samples per node, w, f, r]
                (w is the number of weight sets = the number of conditionals, f = self.config.F)

        """
        assert child_ll is None or (not return_sample_ctx and not with_grad), \
            "If child_ll is provided, return_sample_ctx and with_grad must both be False."
        layer = self.layer_index_to_obj(layer_index)
        assert isinstance(layer, Sum), "Responsibilities can only be computed for Sum layers!"
        ctx = None
        if child_ll is None:
            ctx = self.sample(
                mode='onehot' if with_grad else 'index', n=sample_size, layer_index=layer_index-1, is_mpe=False,
                do_sample_postprocessing=False,
            )
            samples = ctx.sample
            # We sampled all ic nodes of each repetition in the child layer

            child_ll = self.forward(x=samples, layer_index=layer_index-1, x_needs_permutation=False)
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
            w, _, ic, oc, _ = layer.weights.shape
            r = self.config.R
            root_weights_over_rep = layer.weights.view(w, 1, r, ic // r, oc)
            log_weights = th.einsum('wdrio -> wdior', root_weights_over_rep)
            # log_weights = th.log_softmax(root_weights_over_rep, dim=2)
            ll = child_ll.unsqueeze(-2) + log_weights
            ll = th.logsumexp(ll, dim=-3)
        else:
            ll = layer(child_ll)
            log_weights = layer.weights

            # We have the log-likelihood of the current sum layer w.r.t. the samples from its children.
        ll = th.einsum('iwdoR -> wdioR', ll)

        # child_ll now contains the log-likelihood of the samples from all of its 'ic' nodes per feature and
        # repetition - ic * d * r in total.
        # child_ll contains the LL of the samples of each node evaluated among all other nodes - separated
        # by repetition and feature.
        # The tensor shape is [ic, w, d, ic, r]. Looking at one weight set, one feature and one repetition,
        # we are looking at the slice [:, 0, 0, :, 0].
        # The first dimension is the dimension of the samples - there are 'ic' of them.
        # The 4th dimension is the dimension of the LLs of the nodes for those samples.
        # So [4, 0, 0, :, 0] contains the LLs of all nodes given the sample from the fifth node.
        # Likewise, [:, 0, 0, 2, 0] contains the LLs of the samples of all nodes, evaluated at the third node.
        # We needed the full child_ll tensor to compute the LLs of the current layer, but now we only
        # require the LLs of each node's own samples.
        child_ll = child_ll.unsqueeze(-2)
        child_ll = th.einsum('iwdioR -> wdioR', child_ll)

        responsibilities: th.Tensor = log_weights + child_ll - ll
        return responsibilities, (ctx if return_sample_ctx else None)

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

    def natural_param_leaf_update_VIPS(self, samples: th.Tensor, targets: th.Tensor,
                                       step: int, kl_bounds: np.ndarray, verbose=False):
        """
        Updates the mean and log-std parameters of the Gaussian leaves in the natural parameter space.
        As specified in the VIPS paper.

        Args:
            samples: th.Tensor of shape [w, self.config.F, self.config.I, self.config.R, n]
            targets: Tensors of shape [w, self.config.F, self.config.I, self.config.R, n]
        """
        assert samples.shape[1:-1] == (self.config.F, self.config.I, self.config.R)
        samples = samples.cpu()
        targets = targets.cpu()

        with th.no_grad():
            # The PyTorch LS method can deal with high-dim tensors. We want to keep the dimensions intact for
            # as long as possible to ease traceability when debugging.
            design_mat = th.stack([th.ones_like(samples), samples, -0.5 * samples ** 2], dim=-1).cpu()
            last_fit = None
            for i in range(targets.size(0)):
                quad_fit_params, res, rnk, s = th.linalg.lstsq(design_mat, targets[i])
                if last_fit is None:
                    last_fit = quad_fit_params
                elif (((last_fit[..., 1:] - quad_fit_params[..., 1:]) / quad_fit_params[..., 1:]).abs() > 1e-2).any():
                    print(1)

                R = quad_fit_params[..., 2].cpu().numpy().flatten()
                r = quad_fit_params[..., 1].cpu().numpy().flatten()
                c = quad_fit_params[..., 0].cpu().numpy().flatten()
                if False:
                    print(f"Avg. offset: {round(c.mean(), 6)}, "
                          f"avg. linear: {round(r.mean(), 6)}, "
                          f"avg. quadr.: {round(R.mean(), 6)}")

            mu = self.means.cpu().numpy().flatten()
            var = (self.stds ** 2).cpu().numpy().flatten()
        Q = 1 / var
        q = Q * mu
        kl_bounds = kl_bounds.flatten()

        def log_part_fn(X, x):
            return -0.5 * (x ** 2 * 1 / X + np.log(np.abs(2 * np.pi * 1 / X)))

        def Q_q_step(eta, Q, q, R, r):
            Q_step = Q * (eta / (eta + 1)) + R * (1 / (eta + 1))
            if (Q_step < 0.0).any():
                print(2)
            elif ((Q_step - min_Q) < -1e-6).any():
                print(3)
            q_step = q * (eta / (eta + 1)) + r * (1 / (eta + 1))
            return Q_step, q_step

        def loss_fn(eta, Q, q, R, r, eps):
            Q_step, q_step = Q_q_step(eta, Q, q, R, r)
            dual = eta * eps + eta * log_part_fn(Q, q) - (eta + 1) * log_part_fn(Q_step, q_step)
            return dual.sum()

        def KL(eta, Q, q, R, r):
            var = 1 / Q
            mean = var * q
            Q_step, q_step = Q_q_step(eta, Q, q, R, r)
            new_var = 1 / Q_step
            new_mean = new_var * q_step
            KL = 0.5 * (new_var / var + np.log(var / new_var) + (mean - new_mean) ** 2 / var - 1)
            if np.isnan(KL).any():
                print(1)
            return KL

        def grad(*args):
            eps = args[-1]
            return eps - KL(*args[:-1])

        # Calculate bounds for the etas, so that the Q_step are always greater min_Q
        min_Q = 1e-5
        eta_lower_bound = np.abs((min_Q - R) / (Q - min_Q))
        eta_guess = eta_lower_bound + 10
        print(f"Eta mean: {eta_guess.mean()}")

        res = scipy.optimize.minimize(
            loss_fn, eta_guess, args=(Q, q, R, r, kl_bounds),
            # loss_fn, eta_guess[0], args=(Q[0], q[0], R[0], r[0], epsilon[0]),
            method='L-BFGS-B', jac=grad,
            # bounds=scipy.optimize.Bounds(1e-5, np.inf),
            bounds=scipy.optimize.Bounds(eta_lower_bound, np.inf),
        )

        kls = KL(res.x, Q, q, R, r)
        print(f"Max KL div. is {round(kls.max(), 2)}")
        Q_step, q_step = Q_q_step(res.x, Q, q, R, r)
        Q_step = Q_step.reshape(self._leaf.base_leaf.mean_param.shape)
        q_step = q_step.reshape(self._leaf.base_leaf.mean_param.shape)
        new_log_std = -0.5 * np.log(Q_step)
        new_mean = 1 / Q_step * q_step

        def debug_plot(flat_ind, means, samples, eta_lower_bound):
            t = means
            i = flat_ind
            t_ind = flat_index_to_tensor_index(i, t.shape)
            def get(tensor):
                if not isinstance(tensor, th.Tensor):
                    tensor = th.as_tensor(tensor)
                tensor = tensor.detach().clone().cpu()
                if tensor.dim() == 1:
                    tensor = tensor[i]
                else:
                    for ind in t_ind:
                        tensor = th.select(tensor, 0, ind)
                if tensor.dim() == 0:
                    tensor = tensor.item()
                return tensor
            # Is the dual convex? Is the minimum found?
            fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(7, 10), dpi=100)
            fn_args = (Q[i], q[i], R[i], r[i], epsilon[i])
            eta_seq = np.arange(eta_lower_bound[i], 15, 0.1)

            dual = [loss_fn(th.as_tensor(i), *fn_args) for i in eta_seq]
            ax1.plot(eta_seq, dual, label='dual')
            ax1.axvline(res.x[i], color='g',
                        label=f'Min found by L-BFGS-B. KL at min {round(KL(res.x[i], *fn_args[:-1]).item(), 5)}')
            ax1.set_xlabel('eta')
            ax1.legend()

            kls = KL(eta_seq, *fn_args[:-1])
            ax2.plot(eta_seq, kls, label='KL', color='r', linestyle=':')
            ax2.axvline(res.x[i], color='g', label=f'Min found by L-BFGS-B at eta = {round(res.x[i], 2)}.')
            ax2.set_xlabel('eta')
            ax2.legend()

            p = get(quad_fit_params).numpy()
            x = get(samples).numpy()
            for k in kwargs.keys():
                y = get(kwargs[k]).numpy()
                ax3.plot(x, y, 'o', label=k)
            if len(kwargs) > 1:
                y = get(targets).numpy()
                ax3.plot(x, y, 'o', label='Total target')
            xx = np.linspace(np.floor(x.min()), np.ceil(x.max() + 2), 101)
            yy = p[0] + p[1] * xx + p[2] * (-0.5) * xx ** 2
            ax3.plot(xx, yy, label='least squares fit, $y = a + bx - 0.5 * x^2$')
            old_mean = get(mu)
            old_std = np.sqrt(get(var))
            ax3.axvline(old_mean, label=f'old mean, std: {round(old_std, 2)}', color='r',
                        linestyle=':')
            ax3.axvline(get(new_mean), label=f'new mean, std: {round(np.exp(get(new_log_std)), 2)}', color='g')
            ax3.set_xlabel('x')
            ax3.set_ylabel('Reward')
            ax3.legend(framealpha=1, shadow=True, loc=1, fontsize='x-small')
            ax3.grid(alpha=0.25)
            fig.suptitle(f"Optimization metrics of dist param at [{', '.join(np.asarray(t_ind, dtype=str))}]")
            plt.show()
            print(1)

        if verbose:
            t = self.means.data
            flat_ind = 0
            # debug_plot(flat_ind, means=t, samples=samples)
            for i in range(t.numel()):
                debug_plot(i, means=t, samples=samples, eta_lower_bound=eta_lower_bound)
            print(1)

        self.means = new_mean
        self.log_stds = new_log_std
        return res.x

    def reps_weight_update(self, layer_index: int, targets: th.Tensor,
                           reps_instances_of_layer: Dict):
        layer = self.layer_index_to_obj(layer_index)
        layer_log_weights = layer.weights.detach().cpu().numpy()
        _, feat, ic, oc, rep = layer_log_weights.shape
        if layer_index == self.max_layer_index:
            targets = th.einsum('...ior -> ...rio', targets)
            targets = targets.reshape(1, 1, 1, ic, oc, 1)
        targets = targets.detach().cpu().numpy()

        for d in range(feat):
            for c in range(oc):
                for r in range(rep):
                    sel_log_weights = layer_log_weights[0, d, :, c, r]
                    sel_targets = targets[0, 0, d, :, c, r]
                    old_dist = more.Categorical(np.exp(sel_log_weights))
                    reps_inst = reps_instances_of_layer[(d, c, r)][1]
                    kl_bound = reps_instances_of_layer[(d, c, r)][0]
                    new_probabilities = reps_inst.reps_step(kl_bound, -1, old_dist, sel_targets)

                    debug = False
                    if reps_inst.success:
                        layer_log_weights[0, d, :, c, r] = np.log(new_probabilities + 1e-25)
                        new_dist = more.Categorical(new_probabilities)
                        kl = new_dist.kl(old_dist)
                        entropy = new_dist.entropy()
                        if debug:
                            print(f"Layer:{layer_index}, d:{d}, c:{c}, r:{r} - KL new/old {kl:.4f}"
                                  f" - Old ent {old_dist.entropy():.4f} - New ent {entropy:.4f}")
                    elif debug:
                        print(f"Failed for layer:{layer_index}, d:{d}, c:{c}, r:{r}")
        layer.weight_param = nn.Parameter(th.as_tensor(layer_log_weights, device=self.device))

    def more_leaf_update(self, samples: th.Tensor, targets: th.Tensor, more_instances: List[more.MoreGaussian],
                         step: int, kl_bounds: np.ndarray, verbose=False):
        """
        Updates the mean and log-std parameters of the Gaussian leaves in the natural parameter space.
        As specified in the VIPS paper.

        Args:
            samples: th.Tensor of shape [w, self.config.F, self.config.I, self.config.R, n]
            targets: Tensors of shape [w, self.config.F, self.config.I, self.config.R, n]
        """
        samples = samples.clone().cpu().flatten(0, -2).numpy()
        targets = targets.cpu().flatten(1, -2).numpy()
        kl_bounds = kl_bounds.flatten()
        mu = self.means.cpu().numpy().flatten()
        var = (self.stds ** 2).cpu().numpy().flatten()

        res_list = []
        for i in range(len(mu)):
            learner = more_instances[i]
            kl_bound = kl_bounds[i]
            old_mu = np.expand_dims(mu[[i]], -1)
            old_var = np.expand_dims(var[[i]], -1)
            old_dist = more.Gaussian(old_mu, old_var)
            component = more.Gaussian(old_mu, old_var)

            sel_samples = np.expand_dims(samples[i], -1)
            sel_targets = np.expand_dims(targets[:, i], -1)
            for t in sel_targets:
                surrogate = more.QuadFunc(1e-12, normalize=True, unnormalize_output=False)
                surrogate.fit(sel_samples, t, None, old_dist.mean, old_dist.chol_covar)

            # This is a numerical thing we did not use in the original paper: We do not undo the output normalization
            # of the regression, this will yield the same solution but the optimal lagrangian multipliers of the
            # MORE dual are scaled, so we also need to adapt the offset. This makes optimizing the dual much more
            # stable and indifferent to initialization
            if surrogate.o_std is not None:
                learner.eta_offset = 1.0 / np.max((surrogate.o_std, 1e-10))
            else:
                omega = 1.0
                learner.eta_offset = np.abs((learner.min_precision * omega - surrogate.quad_term)
                                            / (1/old_var - learner.min_precision))
                # print(learner.eta_offset)

            new_mean, new_covar = learner.more_step(kl_bound, -1, component, surrogate)
            if learner.success:
                mu[i] = new_mean
                var[i] = new_covar
                res_list.append((1, component.kl(old_dist), component.entropy(), " "))
            else:
                res_list.append((1, 0.0, old_dist.entropy(), "update of component {:d} failed".format(i)))

        param_shape = self.means.shape
        new_mean = mu.reshape(param_shape)
        new_log_std = (0.5 * np.log(var)).reshape(param_shape)
        self.log_stds = new_log_std
        self.means = new_mean
        return res_list

    def layer_entropy_approx(
            self, layer_index=0, child_entropies: th.Tensor = None, child_ll: th.Tensor = None,
            sample_size=5, grad_thru_resp=False, verbose=False,
    ) -> Tuple[th.Tensor, Optional[Dict]]:
        """

        Args:
            layer_index:
            child_entropies: th.Tensor of shape [w, d, oc of child layer == ic of this layer, r]
            child_ll: If provided, children don't need to be sampled. Shape [n, w, d, i, r]
            sample_size:
            grad_thru_resp: If False, the approximation of the responsibility is done without grad.
            verbose: If True, return dict of entropy metrics

        Returns: Tuple of
            node_entropies: th.Tensor of size [w, d, oc of child layer == ic of this layer, r]
            logging: Dict or None, depending on verbose
        """
        assert child_entropies is not None or layer_index == 0, \
            "When sampling from a layer other than the leaf layer, child entropies must be provided!"
        logging = {}
        ctx = None
        aux_responsibility = None

        if layer_index == 0:
            if child_ll is None:
                ctx = self.sample(
                    mode='onehot' if th.is_grad_enabled() else 'index',
                    n=sample_size, layer_index=layer_index, is_mpe=False,
                    do_sample_postprocessing=False,
                )
                # ctx.sample [self.config.I, n, w, self.config.F, self.config.R]
                child_ll = self.forward(
                    x=th.einsum('InwFR -> nwFIR', ctx.sample),
                    layer_index=layer_index, x_needs_permutation=False
                )
            # child_ll [n, w, self.config.F // leaf cardinality, self.config.I, self.config.R]
            node_entropies = -child_ll.mean(dim=0)
        else:
            layer = self.layer_index_to_obj(layer_index)

            if isinstance(layer, CrossProduct):
                node_entropies = layer(child_entropies)
            else:
                with th.set_grad_enabled(grad_thru_resp and th.is_grad_enabled()):
                    aux_responsibility, ctx = self.layer_responsibilities(
                        layer_index=layer_index, sample_size=sample_size, with_grad=grad_thru_resp,
                        child_ll=child_ll
                    )
                    # aux_responsibility [w, d, ic, oc, r]

                [weighted_ch_ents, weighted_aux_resp], weight_entropy = self.weigh_tensors(
                    layer_index=layer_index,
                    tensors=[child_entropies.unsqueeze(3), aux_responsibility],
                    return_weight_ent=True
                )
                node_entropies = weight_entropy + weighted_ch_ents + weighted_aux_resp

                if verbose:
                    metrics = {
                        'weight_entropy': weight_entropy.detach(),
                        'weighted_child_ent': weighted_ch_ents.detach(),
                        'weighted_aux_resp': weighted_aux_resp.detach(),
                    }
                    logging[layer_index] = self.log_dict_from_metric(metrics)

        return node_entropies, logging

    def log_dict_from_metric(self, metrics: Dict, rep_dim=-1, batch_dim: Optional[int] = 0):
        log_dict = {}
        for rep in range(list(metrics.values())[0].size(rep_dim)):
            rep_key = f"rep{rep}"
            rep = th.as_tensor(rep, device=self.device)
            for key, metric in metrics.items():
                if metric is None:
                    continue
                metric = metric.to(self.device)
                log_dict.update({
                    f"{rep_key}/{key}/min": metric.index_select(rep_dim, rep).min().item(),
                    f"{rep_key}/{key}/max": metric.index_select(rep_dim, rep).max().item(),
                    f"{rep_key}/{key}/mean": metric.index_select(rep_dim, rep).mean().item(),
                })
                if batch_dim is not None:
                    log_dict.update({
                        f"{rep_key}/{key}/std": metric.index_select(rep_dim, rep).std(dim=batch_dim).mean().item(),
                    })

        return log_dict

    @staticmethod
    def split_shuffled_scopes(t: th.Tensor, scope_dim: int):
        """
        The scopes in the tensor are split into a left scope and a right scope, where
        the scope dimension contains the shuffled scopes of the left side and the right side,
        [x_{0,L}, x_{0,R}, x_{1,L}, x_{1,R}, x_{2,L}, x_{2,R}, ...]
        """
        shape_before_scope = t.shape[:scope_dim]
        shape_after_scope = t.shape[scope_dim + 1:]
        num_scopes = t.size(scope_dim)
        left_scope_right_scope = t.view(*shape_before_scope, num_scopes // 2, 2, *shape_after_scope)
        dev = left_scope_right_scope.device
        index_dim = scope_dim + 1 if scope_dim > 0 else scope_dim
        left_scope = left_scope_right_scope.index_select(index_dim, th.as_tensor(0, device=dev)).squeeze(index_dim)
        right_scope = left_scope_right_scope.index_select(index_dim, th.as_tensor(1, device=dev)).squeeze(index_dim)
        return left_scope, right_scope

    def vips_sum_prod_pass(self, layer_index: int, log_probs: th.Tensor, accum_weights: th.Tensor):
        if layer_index == self.max_layer_index:
            log_weights = self.root_weights_split_by_rep
        else:
            log_weights = self.layer_index_to_obj(layer_index).weights
        resh_log_weights = self.shape_like_crossprod_input_mapping(log_weights)
        resh_weights = resh_log_weights.exp()
        # weights have shape [w, s, l, r, o, R]
        # log_probs have shape [A, d, n, w, s, o, R], might have additional dim in front
        resh_log_weights = resh_log_weights * accum_weights.unsqueeze(-3).unsqueeze(-3)
        log_probs = resh_log_weights + log_probs.unsqueeze(-3).unsqueeze(-3)
        log_probs = resh_weights * log_probs
        left_scope, right_scope = th.split(log_probs, log_probs.size(-8) // 2, dim=-8)
        left_scope = th.einsum('...AdnwslroR -> ...AdnwsloR', left_scope)
        right_scope = th.einsum('...AdnwslroR -> ...AdnwsroR', right_scope)
        log_probs = th.cat((left_scope, right_scope), dim=-4)
        log_probs = th.einsum('...AdnwsioR -> ...AdnwsiR', log_probs)

        accum_weights = resh_weights * accum_weights.unsqueeze(-3).unsqueeze(-3)
        # resh_weights = th.einsum('wsioR -> wsiR', resh_weights)
        left_weights = th.einsum('wslroR -> wsloR', accum_weights)
        right_weights = th.einsum('wslroR -> wsroR', accum_weights)
        accum_weights = th.cat((left_weights, right_weights), dim=1)
        accum_weights = th.einsum('wsioR -> wsiR', accum_weights)
        return log_probs, accum_weights

    def vips_sum_prod_pass_no_mult(self, layer_index: int, log_probs: th.Tensor, accum_weights: th.Tensor):
        if layer_index == 0:
            log_weights = None
        elif layer_index == self.max_layer_index:
            log_weights = self.root_weights_split_by_rep
        else:
            log_weights = self.layer_index_to_obj(layer_index).weights
        # weights have shape [w, s, l, r, o, R]
        # log_probs have shape [A, d, n, w, s, o, R], might have additional dim in front
        s = log_probs.size(-3)
        if layer_index == 0:
            log_probs = th.einsum('...wsoR -> ...owsR', log_probs)
            log_probs = log_probs.flatten(-5, -4)
        elif s > 1:
            log_probs = log_probs.unsqueeze(-3)
            log_probs = self.shape_like_crossprod_input_mapping(log_probs, -2)
            left_probs, right_probs = th.split(log_probs, s//2, dim=-5)

            left_weights, right_weights = self.split_shuffled_scopes(log_weights, scope_dim=1)
            left_weights = left_weights.unsqueeze(-2)
            right_weights = right_weights.unsqueeze(-3)
            left_probs = left_probs + left_weights
            right_probs = right_probs + right_weights
            log_probs = th.cat((left_probs, right_probs), dim=-5)
            log_probs = log_probs.flatten(-3, -2)
            log_probs = th.einsum('...wsioR -> ...owsiR', log_probs)
            log_probs = log_probs.flatten(-6, -5)
            left_scope, right_scope = th.split(log_probs, log_probs.size(-7) // 2, dim=-7)
            log_probs = th.cat((left_scope, right_scope), dim=-3)
        else:
            log_probs = log_weights + log_probs.unsqueeze(-3)
            log_probs = th.einsum('...wsioR -> ...owsiR', log_probs)
            left_scope, right_scope = th.split(log_probs, log_probs.size(-7) // 2, dim=-7)
            log_probs = th.cat((left_scope, right_scope), dim=-3)

        if log_weights is not None:
            resh_weights = self.shape_like_crossprod_input_mapping(log_weights).exp()
            accum_weights = resh_weights * accum_weights.unsqueeze(-3).unsqueeze(-3)
            # resh_weights = th.einsum('wsioR -> wsiR', resh_weights)
            left_weights = th.einsum('wslroR -> wsloR', accum_weights)
            right_weights = th.einsum('wslroR -> wsroR', accum_weights)
            accum_weights = th.cat((left_weights, right_weights), dim=1)
            accum_weights = th.einsum('wsioR -> wsiR', accum_weights)
        return log_probs, accum_weights

    def vi_entropy_approx(self, sample_size=7, grad_thru_resp: bool = False,
                          verbose: bool = False) -> Tuple[th.Tensor, Optional[Dict]]:
        """
        Args:
            sample_size: Approximate the entropy with this many samples of the leaves.
            grad_thru_resp: If True, entropy approximation can be backpropagated through.
            verbose: Either a bool or a callable that returns bool and takes the current step as its only argument.
                Is evaluated at the beginning of each step to determine if the functions make plots or return logs.

        Returns:

        """
        debug = False
        if debug:
            [self.debug__set_weights_uniform(i) for i in self.sum_layer_indices]
            self.debug__set_dist_params()
            [self.debug__set_weights_dirac(i) for i in self.sum_layer_indices]

        ctx = self.sample(
            mode='onehot' if grad_thru_resp else 'index', n=sample_size, layer_index=0, is_mpe=debug,
            do_sample_postprocessing=False,
        )

        samples = th.einsum('AnwFR -> nwFAR', ctx.sample)
        leaf_ll = self.forward(
            samples, layer_index=0,
        ).mean(0, keepdim=True)
        leaf_entropies = -leaf_ll

        with th.set_grad_enabled(grad_thru_resp):
            ctx = self.sample_postprocessing(ctx=ctx, split_by_scope=True)
            root_ll = self.forward(
                th.einsum('AdnwFR -> AdRnwF', ctx.sample),
                layer_index=self.max_layer_index
            )
            root_ll = th.einsum('AsRnwC -> AsnwCR', root_ll).mean(2, keepdim=True)

        weighed_resp = - root_ll
        weighed_resp = weighed_resp.unsqueeze(-3)

        weighed_weight_ents = None
        accum_weights = th.ones((1, 1, 1, 1), device=root_ll.device)
        for i in reversed(self.sum_layer_indices):
            layer = self.layer_index_to_obj(i)
            layer_weight_ent = -th.sum(layer.weights * layer.weights.exp(), dim=-3)
            # layer_weight_ent = th.mean(layer_weight_ent * accum_weights, dim=-2)
            layer_weight_ent = th.sum(layer_weight_ent * accum_weights, dim=-2)
            if weighed_weight_ents is None:
                weighed_weight_ents = layer_weight_ent
            else:
                weighed_weight_ents = th.repeat_interleave(weighed_weight_ents, 2, dim=1)
                weighed_weight_ents = weighed_weight_ents + layer_weight_ent
            weighed_resp, accum_weights = self.vips_sum_prod_pass(i, weighed_resp, accum_weights)

        weighed_resp = th.einsum('AdnwsAR -> nwsAR', weighed_resp)
        with th.set_grad_enabled(grad_thru_resp):
            weighed_resp = weighed_resp + leaf_ll * accum_weights
        weighed_weight_ents = th.repeat_interleave(weighed_weight_ents, 2, dim=1).unsqueeze(-2)
        weighed_leaf_ents = leaf_entropies * accum_weights
        entropy_approx = weighed_resp + weighed_weight_ents + weighed_leaf_ents
        entropy_approx = th.einsum('nwsAR -> nw', entropy_approx)
        logging = {}
        if verbose:
            metrics = {
                'weight_entropy': weighed_weight_ents.detach(),
                'weighted_child_ent': weighed_leaf_ents.detach(),
                'weighted_resp': weighed_resp.detach(),
            }
            logging = self.log_dict_from_metric(metrics)
        return entropy_approx, logging

    def vi_entropy_approx_bottom_up(self, sample_size=7, grad_thru_resp: bool = False,
                                    verbose: bool = False) -> Tuple[th.Tensor, Optional[Dict]]:
        debug = False
        if debug:
            self.debug__set_dist_params()

        ctx = self.sample(
            mode='onehot' if grad_thru_resp else 'index', n=sample_size, layer_index=0, is_mpe=debug,
            do_sample_postprocessing=False,
        )

        leaf_ll = self.forward(
            th.einsum('AnwFR -> nwFAR', ctx.sample), layer_index=0
        )
        leaf_ll = leaf_ll.mean(0)
        leaf_entropies = -leaf_ll

        weighed_resp = leaf_ll
        weighed_resp = weighed_resp.unsqueeze(0)  # Prepare the scope dim

        accum_weights = th.ones((1, 1, 2, 1, 1))  # [s, w, d, o, R]
        for i in self.sum_layer_indices:
            if i == self.max_layer_index:
                log_weights = self.root_weights_split_by_rep
            else:
                log_weights = self.layer_index_to_obj(i).weights
            # current_weights = current_weights.repeat_interleave(ctx.scopes // d, dim=1)
            resh_log_weights = self.shape_like_crossprod_input_mapping(log_weights)
            resh_weights = resh_log_weights.exp()

            weighed_resp = weighed_resp.unsqueeze(-2)
            weighed_resp_ls, weighed_resp_rs = self.split_shuffled_scopes(weighed_resp, -4)
            weighed_resp_ls = weighed_resp_ls.unsqueeze(-3)
            weighed_resp_rs = weighed_resp_rs.unsqueeze(-4)

            accum_weights = accum_weights.unsqueeze(-2)
            accum_ls, accum_rs = self.split_shuffled_scopes(accum_weights, -4)
            accum_ls = accum_ls.unsqueeze(-3)
            accum_rs = accum_rs.unsqueeze(-4)

            bias_ls = accum_ls * resh_log_weights
            bias_rs = accum_rs * resh_log_weights

            # Unsqueeze for n dim
            weighed_resp_ls = weighed_resp_ls + bias_ls
            weighed_resp_rs = weighed_resp_rs + bias_rs

            weighed_resp_ls = weighed_resp_ls * resh_weights
            weighed_resp_rs = weighed_resp_rs * resh_weights

            weighed_resp_ls = th.einsum('swdlroR -> swdloR', weighed_resp_ls)
            weighed_resp_rs = th.einsum('swdlroR -> swdroR', weighed_resp_rs)
            weighed_resp = th.cat((weighed_resp_ls, weighed_resp_rs), dim=0)
            weighed_resp = th.einsum('swdioR -> swdoR', weighed_resp)

            accum_ls = accum_ls * resh_weights
            accum_rs = accum_rs * resh_weights

            accum_ls = th.einsum('swdlroR -> swdloR', accum_ls)
            accum_rs = th.einsum('swdlroR -> swdroR', accum_rs)
            accum_weights = th.cat((accum_ls, accum_rs), dim=0)
            accum_weights = th.einsum('swdioR -> swdoR', accum_weights)

        assert weighed_resp.size(-3) == 1
        weighed_resp = th.einsum('nswdCR -> nswCR', weighed_resp)

        ctx = self.sample_postprocessing(ctx=ctx, split_by_scope=True)
        root_ll = self.forward(
            th.einsum('AdnwFR -> AdRnwF', ctx.sample),
            layer_index=self.max_layer_index
        )
        root_ll = th.einsum('AsRnwC -> AsnwCR', root_ll)


        accum_weights = th.einsum('ABsnwdCR -> ABsnwCR', accum_weights.unsqueeze(3))
        log_probs = - root_ll * accum_weights + log_probs

        layer_log = {}
        return vips_logging

    def vips(self, target_dist_callback, steps, step_callback: Callable = None,
             sample_size=7, init_leaf_kl_bound=1e-2, init_weight_kl_bound=1e-2, weight_update_start=0,
             verbose: Union[bool, Callable] = False) -> Dict:
        """

        Args:
            target_dist_callback: Callback to return a th.Tensor of log probabilities of the samples
                on the target distribution.
            steps:
            step_callback: Function that is called at the end of each step with the current step as the arg.
            sample_size:
            verbose: Either a bool or a callable that returns bool and takes the current step as its only argument.
                Is evaluated at the beginning of each step to determine if the functions make plots or return logs.

        Returns:

        """
        [self.debug__set_weights_uniform(i) for i in self.sum_layer_indices]
        debug = False
        if debug:
            self.debug__set_dist_params()
            [self.debug__set_weights_dirac(i) for i in self.sum_layer_indices]
        vips_logging = {}
        grad_thru_resp = False

        more_instances = [more.MoreGaussian(1, 1.0, 1.0, False) for _ in range(self.means.numel())]
        weight_learners = {}
        for l in self.sum_layer_indices:
            weight_learners[l] = {}
            _, feat, _, oc, rep = self.layer_index_to_obj(l).weight_param.shape
            for d in range(feat):
                for c in range(oc):
                    for r in range(rep):
                        weight_learners[l].update({
                            (d, c, r): (init_weight_kl_bound, more.RepsCategorical(1.0, 0.0, False)),
                        })

        leaf_kl_bounds = init_leaf_kl_bound * np.ones(self._leaf.base_leaf.mean_param.shape)

        for step in range(int(steps)):
            verbose_on = verbose(step) if callable(verbose) else verbose
            etas = None

            layer_entropies = None
            for layer_index in self.sum_layer_indices:
                layer_log = {}
                layer = self.layer_index_to_obj(layer_index)

                ctx = self.sample(
                    mode='index', n=sample_size, layer_index=layer_index-2, is_mpe=debug,
                    do_sample_postprocessing=False,
                )

                sampled_self_ll = self.forward(ctx.sample, layer_index=layer_index-2)
                child_ll = th.einsum('AnwdAR -> nwdAR', sampled_self_ll)
                sampled_self_ll = sampled_self_ll.mean(1)
                if layer_index == 2:
                    sampled_node_entropies, ent_log = self.layer_entropy_approx(
                        layer_index=layer_index-2,
                        child_ll=child_ll,
                        verbose=True,
                    )
                    vips_logging.update({layer_index-2: ent_log})
                else:
                    sampled_node_entropies = layer_entropies
                    child_ll = child_ll.mean(0, keepdim=True)

                prod_entropies, _ = self.layer_entropy_approx(
                    layer_index=layer_index-1,
                    child_entropies=sampled_node_entropies,
                )
                ic, w, d_old, ic, R = sampled_self_ll.shape
                prod_ll_ls, prod_ll_rs = self.split_shuffled_scopes(sampled_self_ll, 2)
                prod_ll_ls = prod_ll_ls.unsqueeze(4).unsqueeze(1)
                prod_ll_rs = prod_ll_rs.unsqueeze(3).unsqueeze(0)
                prod_ll = prod_ll_ls + prod_ll_rs
                prod_ll = prod_ll.view(ic ** 2, w, d_old // 2, ic ** 2, R)
                layer_entropies, ent_log = self.layer_entropy_approx(
                    layer_index=layer_index,
                    child_ll=prod_ll,
                    child_entropies=prod_entropies,
                    verbose=True,
                )
                vips_logging.update({layer_index: ent_log})

                ctx = self.sample_postprocessing(ctx=ctx, split_by_scope=True)
                root_ll = self.forward(th.einsum('AdnwFR -> AdRnwF', ctx.sample),
                                       layer_index=self.max_layer_index)
                samples = th.einsum('AdnwFR -> AdRnwF', ctx.sample)
                if target_dist_callback is None:
                    target_ll = th.zeros_like(root_ll)
                else:
                    target_ll = target_dist_callback(samples)
                    # only for vips_gmm:
                    target_ll = target_ll.nan_to_num().sum(-1, keepdim=True).to(root_ll.device)
                if layer_index > 2:
                    root_ll = root_ll.mean(-3, keepdim=True)
                    target_ll = target_ll.mean(-3, keepdim=True)

                if False:
                    biased_target = target_ll + 200.0
                    target_ll = th.stack((target_ll, biased_target), dim=0)
                else:
                    # In natural_param_leaf_update this dim is iterated over
                    target_ll = target_ll.unsqueeze(0)

                root_ll = th.einsum('AsRnwC -> AsnwCR', root_ll)
                target_ll = th.einsum('...AsRnwC -> ...AsnwCR', target_ll)
                root_target_diff = target_ll - root_ll
                log_probs = root_target_diff
                log_probs = log_probs.unsqueeze(-3)

                use_no_mult = True

                accum_weights = th.ones((1, 1, 1, 1), device=root_ll.device)
                if use_no_mult:
                    for i in reversed([0, *self.sum_layer_indices[(layer_index//2-1):]]):
                        log_probs, accum_weights = self.vips_sum_prod_pass_no_mult(i, log_probs, accum_weights)
                    log_probs = log_probs.mean(4)
                else:
                    for i in reversed(self.sum_layer_indices[(layer_index//2):]):
                        log_probs, accum_weights = self.vips_sum_prod_pass(i, log_probs, accum_weights)

                if layer_index == 2:
                    samples = ctx.sample.nan_to_num()
                    samples = th.einsum('AsnwFR -> wFARn', samples)

                    if use_no_mult:
                        leaf_log_probs = th.einsum('...AdnwsR -> ...nwsAR', log_probs)
                        leaf_log_probs = leaf_log_probs + child_ll
                    else:
                        leaf_log_probs, leaf_accum_weights = self.vips_sum_prod_pass(layer_index, log_probs, accum_weights)
                        leaf_log_probs = th.einsum('...AdnwsAR -> ...nwsAR', leaf_log_probs)
                        leaf_log_probs = leaf_log_probs + (child_ll * leaf_accum_weights)
                    leaf_log_probs = leaf_log_probs.repeat_interleave(self._leaf.cardinality, dim=-3)
                    leaf_log_probs = th.einsum('...nwsAR -> ...wsARn', leaf_log_probs)
                    leaf_param_args = {
                        'samples': samples,
                        'verbose': verbose_on,
                        'targets': leaf_log_probs,
                        'step': step,
                        'kl_bounds': leaf_kl_bounds,
                    }
                    with th.no_grad():
                        if False:
                            res = self.natural_param_leaf_update_VIPS(
                                **leaf_param_args,
                            )
                        else:
                            res = self.more_leaf_update(
                                more_instances=more_instances,
                                **leaf_param_args,
                            )

                if step < weight_update_start:
                    continue
                child_ll = child_ll.mean(0)
                if use_no_mult:
                    log_probs = log_probs.mean(-4)
                    left_scope, right_scope = th.split(log_probs, log_probs.size(-2) // 2, dim=-2)
                    left_scope = left_scope.unsqueeze(-5)
                    right_scope = right_scope.unsqueeze(-6)
                    log_probs = left_scope + right_scope
                    log_probs = log_probs.flatten(1, 2)
                    log_probs = th.einsum('...idwoR -> ...wdioR', log_probs)
                else:
                    log_probs = log_probs.mean(-5)
                    prod_layer = self.layer_index_to_obj(layer_index - 1)
                    # We pass the biased target LLs through the child product layer.
                    left_scope, right_scope = th.split(log_probs, log_probs.size(-5) // 2, dim=-5)
                    log_probs = th.cat((left_scope, right_scope), dim=-3)
                    log_probs = log_probs.squeeze(-5)
                    log_probs = th.einsum('...AwdoR -> ...owdAR', log_probs)
                    log_probs = prod_layer.forward(log_probs)
                    log_probs = th.einsum('...owdiR -> ...wdioR', log_probs)

                    # To complete the total-SPN-responsibility, we add the child product nodes' LLs and the
                    # log weights of this layer, both weighed by the accumulated weights
                    if layer_index == self.max_layer_index:
                        log_weights = self.root_weights_split_by_rep
                    else:
                        log_weights = self.layer_index_to_obj(layer_index).weights
                    biased_prod_ll = prod_layer.forward(child_ll).unsqueeze(-2) + log_weights
                    biased_prod_ll = biased_prod_ll * accum_weights.unsqueeze(-3)
                    weighed_prod_ent = prod_entropies.unsqueeze(-2) * accum_weights.unsqueeze(-3)

                    log_probs = log_probs + biased_prod_ll + weighed_prod_ent

                if init_weight_kl_bound > 0.0:
                    weight_param_args = {
                        'targets': log_probs,
                        'layer_index': layer_index,
                    }
                    self.reps_weight_update(reps_instances_of_layer=weight_learners[layer_index], **weight_param_args)
                else:
                    new_weights = log_probs
                    if layer_index == self.max_layer_index:
                        ic, oc, r = new_weights.shape[-3:]
                        new_weights = th.einsum('...ior -> ...roi', new_weights)
                        new_weights = new_weights.reshape(*new_weights.shape[:-3], ic * r, oc, 1)
                        # new_weights [w, 1, ic * r, 1, 1]
                    new_weights = th.stack([F.log_softmax(t, dim=-3) for t in new_weights], dim=0)
                    layer.weight_param = nn.Parameter(new_weights[0])

                vips_logging.update({layer_index: layer_log})
            if step_callback is not None:
                step_callback(step)
        return vips_logging

    def vips_bottom_up(self, target_dist_callback, steps, step_callback: Callable = None,
                                 sample_size=7, verbose: Union[bool, Callable] = False) \
            -> Dict:
        """

        Args:
            target_dist_callback: Callback to return a th.Tensor of log probabilities of the samples
                on the target distribution.
            steps:
            step_callback: Function that is called at the end of each step with the current step as the arg.
            sample_size:
            verbose: Either a bool or a callable that returns bool and takes the current step as its only argument.
                Is evaluated at the beginning of each step to determine if the functions make plots or return logs.

        Returns:

        """
        debug = False
        if debug:
            self.debug__set_dist_params()
        vips_logging = {}
        grad_thru_resp = False

        for step in range(int(steps)):
            verbose_on = verbose(step) if callable(verbose) else verbose
            etas = None

            log_probs = self_ll = None
            layer_entropies = None
            for layer_index in self.sum_layer_indices:
                layer_log = {}
                layer = self.layer_index_to_obj(layer_index)

                ctx = self.sample(
                    mode='index', n=sample_size, layer_index=layer_index-2, is_mpe=debug,
                    do_sample_postprocessing=False,
                )
                # We can get the self_ll simply by permuting the samples to line up with the dist parameters
                sampled_self_ll = self.forward(ctx.sample, layer_index=layer_index-2)
                child_ll = th.einsum('AnwdAR -> nwdAR', sampled_self_ll)
                sampled_self_ll = sampled_self_ll.mean(1)
                if layer_index == 2:
                    sampled_node_entropies, ent_log = self.layer_entropy_approx(
                        layer_index=layer_index-2,
                        child_ll=child_ll,
                        verbose=True,
                    )
                    vips_logging.update({layer_index-2: ent_log})
                else:
                    sampled_node_entropies = layer_entropies
                    child_ll = child_ll.mean(0, keepdim=True)

                prod_entropies, _ = self.layer_entropy_approx(
                    layer_index=layer_index-1,
                    child_entropies=sampled_node_entropies,
                )
                ic, w, d_old, ic, R = sampled_self_ll.shape
                prod_ll_ls, prod_ll_rs = self.split_shuffled_scopes(sampled_self_ll, 2)
                prod_ll_ls = prod_ll_ls.unsqueeze(4).unsqueeze(1)
                prod_ll_rs = prod_ll_rs.unsqueeze(3).unsqueeze(0)
                prod_ll = prod_ll_ls + prod_ll_rs
                prod_ll = prod_ll.view(ic ** 2, w, d_old // 2, ic ** 2, R)
                layer_entropies, ent_log = self.layer_entropy_approx(
                    layer_index=layer_index,
                    child_ll=prod_ll,
                    child_entropies=prod_entropies,
                    verbose=True,
                )
                vips_logging.update({layer_index: ent_log})

                log_probs = child_ll  # target_ll - root_ll + child_ll
                log_probs = log_probs.unsqueeze(0)  # Prepare the scope dim
                log_probs = log_probs.unsqueeze(0)  # Prepare the dim of nodes of the first sum layer
                log_probs = log_probs.unsqueeze(0)  # Prepare the dim over sampled nodes

                accum_resh_weights = accum_ls = accum_rs = None
                for i in self.sum_layer_indices[(layer_index//2-1):]:
                    if i == self.max_layer_index:
                        log_weights = self.root_weights_split_by_rep
                    else:
                        log_weights = self.layer_index_to_obj(i).weights
                    # current_weights = current_weights.repeat_interleave(ctx.scopes // d, dim=1)
                    resh_log_weights = self.shape_like_crossprod_input_mapping(log_weights)
                    resh_weights = resh_log_weights.exp()

                    # Previous layer's oc is current layer's ic
                    log_probs = log_probs.unsqueeze(-2)
                    left_scope, right_scope = self.split_shuffled_scopes(log_probs, -4)

                    # Unsqueeze to create dim to sum scope-split weights over
                    left_scope = left_scope.unsqueeze(-3)
                    right_scope = right_scope.unsqueeze(-4)

                    if accum_resh_weights is None:
                        left_scope = left_scope + resh_log_weights
                        right_scope = right_scope + resh_log_weights
                    else:
                        accum_ls, accum_rs = self.split_shuffled_scopes(accum_resh_weights.unsqueeze(-2), -4)

                        accum_ls = accum_ls.unsqueeze(-3)
                        weighted_ls_log_w = accum_ls * resh_log_weights

                        accum_rs = accum_rs.unsqueeze(-4)
                        weighted_rs_log_w = accum_rs * resh_log_weights

                        # Unsqueeze for n dim
                        left_scope = weighted_ls_log_w.unsqueeze(3) + left_scope
                        right_scope = weighted_rs_log_w.unsqueeze(3) + right_scope

                    left_scope = left_scope * resh_weights
                    left_scope = th.einsum('ABsnwdlroR -> ABsnwdloR', left_scope)
                    right_scope = right_scope * resh_weights
                    right_scope = th.einsum('ABsnwdlroR -> ABsnwdroR', right_scope)

                    log_probs = th.cat((left_scope, right_scope), dim=2)

                    if i == layer_index:
                        resh_weights_ls = th.einsum('wdlroR -> wdloR', resh_weights)
                        resh_weights_rs = th.einsum('wdlroR -> wdroR', resh_weights)
                        accum_resh_weights = th.stack((resh_weights_ls, resh_weights_rs), dim=0)
                        accum_resh_weights = accum_resh_weights.unsqueeze(0)
                        accum_resh_weights = th.einsum('BswdAoR -> ABswdoR', accum_resh_weights)

                        log_probs = log_probs.squeeze(0)
                        log_probs = th.einsum('BsnwdAoR -> ABsnwdoR', log_probs)
                    else:
                        accum_ls = accum_ls * resh_weights
                        accum_ls = th.einsum('ABswdlroR -> ABswdloR', accum_ls)
                        accum_rs = accum_rs * resh_weights
                        accum_rs = th.einsum('ABswdlroR -> ABswdroR', accum_rs)
                        accum_resh_weights = th.cat((accum_ls, accum_rs), dim=2)

                        if i == layer_index + 2:
                            # put current ic dim with the batch dims so we have something to update the sum weights with
                            log_probs = log_probs.squeeze(1)
                            log_probs = th.einsum('AsnwdBoR -> ABsnwdoR', log_probs)
                            accum_resh_weights = accum_resh_weights.squeeze(1)
                            accum_resh_weights = th.einsum('AswdBoR -> ABswdoR', accum_resh_weights)
                        else:
                            # Sum over ic dim
                            log_probs = th.einsum('ABsnwdioR -> ABsnwdoR', log_probs)
                            accum_resh_weights = th.einsum('ABswdioR -> ABswdoR', accum_resh_weights)

                assert log_probs.size(-3) == 1
                log_probs = th.einsum('ABsnwdCR -> ABsnwCR', log_probs)

                ctx = self.sample_postprocessing(ctx=ctx, split_by_scope=True)
                samples = th.einsum('AdnwFR -> AdRnwF', ctx.sample)
                root_ll = self.forward(samples, layer_index=self.max_layer_index)
                root_ll = root_ll.unsqueeze(1)
                if target_dist_callback is None:
                    target_ll = th.zeros_like(root_ll)
                else:
                    target_ll = target_dist_callback(samples)
                    # only for vips_gmm:
                    target_ll = target_ll.unsqueeze(1).nan_to_num().sum(-1, keepdim=True).to(root_ll.device)
                    # target_ll += 200.0

                root_ll = th.einsum('ABsRnwC -> ABsnwCR', root_ll)
                target_ll = th.einsum('ABsRnwC -> ABsnwCR', target_ll)

                accum_resh_weights = th.einsum('ABsnwdCR -> ABsnwCR', accum_resh_weights.unsqueeze(3))
                log_probs = (target_ll - root_ll) * accum_resh_weights + log_probs
                if layer_index == self.max_layer_index:
                    log_probs = log_probs.squeeze(1)
                    log_probs = th.einsum('AsnwBR -> ABsnwR', log_probs)
                else:
                    log_probs = th.einsum('ABsnwCR -> ABsnwR', log_probs)

                if layer_index == 2:
                    samples = ctx.sample.nan_to_num()
                    samples = th.einsum('AsnwFR -> wFARn', samples)

                    log_probs_for_update = th.einsum('ABsnwR -> wsARn', log_probs)
                    log_probs_for_update = log_probs_for_update.repeat_interleave(self._leaf.cardinality, dim=1)
                    etas = self.natural_param_leaf_update_VIPS(
                        samples=samples,
                        eta_guess=etas,
                        verbose=verbose_on,
                        targets=log_probs_for_update.unsqueeze(0),
                    )

                continue
                log_probs = log_probs.mean(dim=3)
                accum_resh_weights = th.einsum('ABsnwCR -> wsABR', accum_resh_weights)
                weighed_ch_ent = sampled_node_entropies.unsqueeze(-2) * accum_resh_weights
                log_probs = th.einsum('ABswR -> wsABR', log_probs)
                log_probs = log_probs + weighed_ch_ent
                left_scope, right_scope = self.split_shuffled_scopes(log_probs, scope_dim=1)
                left_scope = left_scope.unsqueeze(2)
                right_scope = right_scope.unsqueeze(3)
                new_weights = left_scope + right_scope
                new_weights = new_weights.flatten(2, 3)
                if layer_index == self.max_layer_index:
                    w, d, ic, oc, r = new_weights.shape
                    new_weights = th.einsum('...ior -> ...roi', new_weights)
                    new_weights = new_weights.reshape(w, d, ic * r, oc, 1)
                new_weights = F.log_softmax(new_weights, dim=2)
                # new_weights [w, 1, ic * r, 1, 1]
                layer.weight_param = nn.Parameter(new_weights)

                vips_logging.update({layer_index: layer_log})
            if step_callback is not None:
                step_callback(step)
        return vips_logging

    def vi_entropy_approx_layerwise(self, sample_size=10, grad_thru_resp: bool = False, verbose=False) \
            -> Tuple[th.Tensor, Optional[Dict]]:
        """
        Approximate the entropy of the root sum node via variational inference,
        as done in the Variational Inference by Policy Search paper.

        Args:
            sample_size: Number of samples to approximate the expected entropy of the responsibility with.
            grad_thru_resp: If False, the approximation of the responsibility is done without grad.
            verbose: Return logging data
        """
        logging = {}
        child_entropies = None
        for layer_index in range(self.num_layers):
            child_entropies, layer_log = self.layer_entropy_approx(
                layer_index=layer_index, child_entropies=child_entropies,
                sample_size=sample_size, grad_thru_resp=grad_thru_resp,
                verbose=verbose,
            )
            logging.update({layer_index: layer_log})
        return child_entropies.flatten(), logging

    def monte_carlo_ent_approx(self, sample_size=100, layer_index: int = None, sample_with_grad=False):
        if layer_index is None:
            layer_index = self.max_layer_index

        sample_args = {
            'n': sample_size, 'layer_index': layer_index, 'do_sample_postprocessing': False
        }
        if sample_with_grad:
            samples = self.sample(mode='onehot', **sample_args)
        else:
            with th.no_grad():
                samples = self.sample(mode='index', **sample_args)

        log_probs = self.forward(th.einsum('o...FR -> ...FoR', samples.sample), layer_index=layer_index)
        return -log_probs.mean()

    def huber_entropy_lb(self, layer_index: int = None, verbose=True):
        """
        Calculate the entropy lower bound of the SPN as in Huber '08.

        Args:
            layer_index: Compute the entropy LB for this layer. Can take values from 1 to self.max_layer_index+1.
                self.max_layer_index+1 is the sampling root.

        Returns:

        """
        logging = {}
        entropy_lb = None
        if layer_index is None:
            layer_index = self.max_layer_index

        # Distribution layer of leaf layer
        leaf = self._leaf.base_leaf
        means = leaf.inv_permutation_means
        var = leaf.inv_permutation_log_stds.exp() ** 2
        var = var.unsqueeze(2).unsqueeze(-2) + var.unsqueeze(3).unsqueeze(-1)
        std = th.sqrt(var)
        gauss = dist.Normal(means.unsqueeze(2).unsqueeze(-2).expand_as(std), std)
        log_probs = gauss.log_prob(means.unsqueeze(3).unsqueeze(-1).expand_as(std))
        permutation = leaf.permutation.unsqueeze(-2).unsqueeze(-2).expand_as(log_probs)
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
                left_scope, right_scope = self.split_shuffled_scopes(log_probs, 1)
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
                if i < self.max_layer_index:
                    weights = self.layer_index_to_obj(i).weights
                elif i == self.max_layer_index:
                    weights = self.root_weights_split_by_rep
                else:
                    weights = self._sampling_root.weights

                # weights = weights.unsqueeze(2).unsqueeze(-3).unsqueeze(-2) + weights.unsqueeze(3).unsqueeze(-2).unsqueeze(-1)
                # log_probs = log_probs.unsqueeze(-3).unsqueeze(-3) + weights
                # log_probs = log_probs.logsumexp(2).logsumexp(2)
                log_probs = log_probs.unsqueeze(-3) + weights.unsqueeze(2).unsqueeze(-2)
                # We don't logsumexp over the unsqueezed dims
                log_probs = log_probs.logsumexp(3)
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

    def set_all_consolidated_weights(self):
        for i in self.sum_layer_indices:
            self.set_layer_consolidated_weights(i)

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
