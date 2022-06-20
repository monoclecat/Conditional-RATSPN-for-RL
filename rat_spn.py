import logging
from typing import Dict, Type, List, Union, Optional, Tuple, Callable
import math

import numpy as np
import torch as th
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from torch import distributions as dist
import scipy
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

    def _randomize(self, x: th.Tensor) -> th.Tensor:
        """
        Randomize the input at each repetition according to `self.permutation`.

        Args:
            x: Input.

        Returns:
            th.Tensor: Randomized input along feature axis. Each repetition has its own permutation.
        """
        assert x.size(-1) == 1 or x.size(-1) == self.config.R
        if x.size(-1) == 1:
            x = x.repeat(*([1] * (x.dim()-1)), self.config.R)
        perm_indices = self.permutation.expand_as(x)
        x = th.gather(x, dim=-2, index=perm_indices)

        return x

    def forward(self, x: th.Tensor, layer_index: int = None, x_needs_permutation: bool = True) -> th.Tensor:
        """
        Forward pass through RatSpn. Computes the conditional log-likelihood P(X | C).

        Args:
            x:
                Input of shape [*batch_dims, weight_sets, self.config.F,
                                self.config.R or 1 if no rep is to be specified].
                    batch_dims: Sample shape per weight set (= per conditional in the CSPN sense).
                    weight_sets: In CSPNs, weights are different for each conditional. In RatSpn, this is 1.
            layer_index: Evaluate log-likelihood of x at layer
            x_needs_permutation: An SPNs own samples where no inverted permutation was applied, don't need to be
                permuted in the forward pass.
        Returns:
            th.Tensor: Conditional log-likelihood P(X | C) of the input.
        """
        if layer_index is None:
            layer_index = self.max_layer_index

        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.size(-1) == self.config.F:
            # x has no repetition dimension
            x = x.unsqueeze(-1)

        if x_needs_permutation:
            # Apply feature randomization for each repetition
            x = self._randomize(x)

        # Apply leaf distributions
        x = self._leaf(x)

        # Forward to inner product and sum layers
        for layer in self._inner_layers[:layer_index]:
            x = layer(x)

        if layer_index == self.max_layer_index:
            # Merge results from the different repetitions into the channel dimension
            w, d, c, r = x.shape[-4:]
            batch_dims = x.shape[:-4]
            assert d == 1  # number of features should be 1 at this point
            x = x.view(*batch_dims, w, d, c * r, 1)

            # Apply C sum node outputs
            x = self.root(x)

            # Remove repetition dimension
            x = x.squeeze(-1)

            # Remove in_features dimension
            x = x.squeeze(-3)

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
            th.ones(size=(1, self.config.C, 1, 1)) * th.tensor(1 / self.config.C), requires_grad=False
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
                    ctx.parent_indices = th.as_tensor(class_index).repeat_interleave(n)
                    if mode == 'onehot':
                        ctx.parent_indices = F.one_hot(ctx.parent_indices, num_classes=self.config.C)
                    ctx.n = n * len(class_index)
                    ctx.parent_indices = ctx.parent_indices.unsqueeze(1).unsqueeze(1).unsqueeze(0).unsqueeze(-1)

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

                # The weights of the root sum node represent the input channel and repetitions in this manner:
                # The CSPN case is assumed where the weights are different for each batch index condition.
                # Looking at one batch index and one output channel, there are IC*R weights.
                # An element of this weight vector is defined as
                # w_{r,c}, with r and c being the repetition and channel the weight belongs to, respectively.
                # The weight vector will then contain [w_{0,0},w_{1,0},w_{2,0},w_{0,1},w_{1,1},w_{2,1},w_{0,2},...]
                # This weight vector was used as the logits in a IC*R-categorical distribution,
                # yielding indexes [0,C*R-1].
                if mode == 'index':
                    # ctx.parent_indices = ctx.parent_indices.squeeze(-1)
                    # To match the index to the correct repetition and its input channel, we do the following
                    ctx.repetition_indices = (ctx.parent_indices % self.config.R).squeeze(-1).squeeze(-1)
                    # repetition_indices [nr_nodes, *sample_shape, w]
                    ctx.parent_indices = th.div(ctx.parent_indices, self.config.R, rounding_mode='trunc')
                    # parent_indices [nr_nodes, *sample_shape, w, d, r = 1]
                else:
                    ctx.parent_indices = ctx.parent_indices.view(*ctx.parent_indices.shape[:-2], -1, self.config.R)
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
                samples = samples.permute(3, 0, 1, 2, 4)
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
            assert self.config.F % ctx.scopes == 0, "Size of entire scope is not divisible by the number of" \
                                                    " scopes in the layer we are sampling from. What do?"
            features_per_scope = self.config.F // ctx.scopes

            sample = sample.unsqueeze(1)
            sample = sample.repeat(1, ctx.scopes, *([1] * (sample.dim()-2)))

            for i in range(ctx.scopes):
                mask = th.ones(self.config.F, dtype=th.bool)
                mask[i * features_per_scope:(i + 1) * features_per_scope] = False
                sample[:, i, :, :, mask, ...] = th.nan

        # Each repetition has its own inverted permutation which we now apply to the samples.
        if invert_permutation:
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

            if split_by_scope:
                rep_sel_inv_perm = rep_sel_inv_perm.unsqueeze(1)
            rep_sel_inv_perm = rep_sel_inv_perm.expand_as(sample)
            sample = th.gather(sample, dim=-2 if ctx.has_rep_dim else -1, index=rep_sel_inv_perm)
            ctx.permutation_inverted = True

        if self.config.tanh_squash:
            sample = sample.clamp(-6.0, 6.0).tanh()
            ctx.needs_squashing = False

        if not ctx.has_rep_dim:
            sample = sample.unsqueeze(-1)

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
        ctx.parent_indices = None
        ctx.repetition_indices = None
        return ctx

    def sample_index_style(self, **kwargs):
        return self.sample(mode='index', **kwargs)

    def sample_onehot_style(self, **kwargs):
        return self.sample(mode='onehot', **kwargs)

    def approx_responsibilities(self, layer_index, sample_size: int = 5, return_sample_ctx: bool = False,
                                with_grad=False) \
            -> Tuple[th.Tensor, Optional[Sample]]:
        """
        Approximate the responsibilities of a Sum layer in the Spn.
        For this it draws samples of the child nodes of the layer. These samples can be returned as well.

        Args:
            layer_index: The layer index to evaluate. The layer must be a Sum layer.
            sample_size: Number of samples to evaluate responsibilities on.
                Responsibilities can't be computed in closed form.
            return_sample_ctx: If True, the sample context including the samples are returned. No post-processing is
                applied to these.
            with_grad: If True, sampling is done in a differentiable way.

        Returns: Tuple of
            responsibilities: th.Tensor of shape [w, d, oc of child layer, oc, r of child layer].
            samples: th.Tensor of shape [ic, samples per node, w, f, r]
                (w is the number of weight sets = the number of conditionals, f = self.config.F)

        """
        layer = self.layer_index_to_obj(layer_index)
        assert isinstance(layer, Sum), "Responsibilities can only be computed for Sum layers!"
        ctx = self.sample(
            mode='onehot' if with_grad else 'index', n=sample_size, layer_index=layer_index-1, is_mpe=False,
            do_sample_postprocessing=False,
        )
        samples = ctx.sample
        # child_samples [oc of current layer, sample_size, w, self.config.F, r]

        # We sampled all ic nodes of each repetition in the child layer
        ic, n, w, f, r = samples.shape

        child_ll = self.forward(x=samples, layer_index=layer_index-1, x_needs_permutation=False)
        child_ll = child_ll.mean(dim=1)

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
            # We cannot simply reshape aux_responsibility to the shape of the weights and renormalize
            # because the responsibilities are only valid within the repetition where they are calculated.
            # So we need to reshape and normalize the root weights to imitate sum nodes for each
            # repetition separately, and then a sum node over the three repetitions.
            # The same calculations that are applied to the aux_responsibilities must also be applied to
            # the child entropies. Otherwise, we mess up the gradients.
            weights = weights.view(*weights.shape[:2], self.config.R, -1).permute(0, 1, 3, 2).unsqueeze(-2)
            # weights [w, d, ic, oc=1, self.config.R]

            weights_over_ic = weights.softmax(dim=2)
            tensors = [th.sum(weights_over_ic * t, dim=2) for t in tensors]

            weights_over_rep = weights.sum(dim=2).softmax(dim=-1)
            tensors = [th.sum(weights_over_rep * t, dim=-1, keepdim=True) for t in tensors]

        return tensors, weight_entropy

    def natural_param_leaf_update(self, samples: th.Tensor, eta_guess: np.ndarray, verbose=False, **kwargs):
        """
        Updates the mean and log-std parameters of the Gaussian leaves in the natural parameter space.
        As specified in the VIPS paper.

        Args:
            samples: th.Tensor of shape [w, self.config.F, self.config.I, self.config.R, n]
            targets: Tensors of shape [w, self.config.F, self.config.I, self.config.R, n]
        """
        assert samples.shape[1:-1] == (self.config.F, self.config.I, self.config.R)
        samples = samples.clone().cpu()
        keys = list(kwargs.keys())
        for k in keys:
            if kwargs[k] is None or not isinstance(kwargs[k], th.Tensor):
                kwargs.pop(k)
                continue
            assert kwargs[k].shape == (samples.shape[0], self.config.F, self.config.I, self.config.R, samples.shape[-1])
            kwargs[k] = kwargs[k].cpu()

        targets = th.stack([val for val in kwargs.values() if val is not None], dim=0).sum(dim=0)

        with th.no_grad():
            # The PyTorch LS method can deal with high-dim tensors. We want to keep the dimensions intact for
            # as long as possible to ease traceability when debugging.
            design_mat = th.stack([th.ones_like(samples), samples, -0.5 * samples ** 2], dim=-1).cpu()
            quad_fit_params, res, rnk, s = th.linalg.lstsq(design_mat, targets)
            R = quad_fit_params[..., 2].numpy().flatten()
            r = quad_fit_params[..., 1].numpy().flatten()

            mu = self._leaf.base_leaf.mean_param.cpu().numpy().flatten()
            var = (self._leaf.base_leaf.std_param.exp() ** 2).cpu().numpy().flatten()
        Q = 1 / var
        q = Q * mu

        epsilon = np.ones_like(Q) * 0.01

        def log_part_fn(X, x):
            return -0.5 * (x ** 2 * 1 / X + np.log(np.abs(2 * np.pi * 1 / X)))

        def Q_q_step(eta, Q, q, R, r):
            Q_step = Q * (eta / (eta + 1)) + R * (1 / (eta + 1))
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
            return KL

        def grad(*args):
            eps = args[-1]
            return eps - KL(*args[:-1])

        # Calculate bounds for the etas, so that the Q_step are always greater min_Q
        min_Q = 1e-5
        eta_lower_bound = (min_Q - R) / (Q - min_Q)
        eta_lower_bound = np.max((eta_lower_bound, np.zeros_like(eta_lower_bound)), axis=0)
        if eta_guess is None:
            eta_guess = eta_lower_bound * 10
        print(f"Eta mean: {eta_guess.mean()}")

        res = scipy.optimize.minimize(
            loss_fn, eta_guess, args=(Q, q, R, r, epsilon),
            # loss_fn, eta_guess[0], args=(Q[0], q[0], R[0], r[0], epsilon[0]),
            method='L-BFGS-B', jac=grad, bounds=scipy.optimize.Bounds(eta_lower_bound, np.inf),
        )

        kls = KL(res.x, Q, q, R, r)
        print(f"Max KL div. is {round(kls.max(), 2)}")
        Q_step, q_step = Q_q_step(res.x, Q, q, R, r)
        Q_step = Q_step.reshape(self._leaf.base_leaf.mean_param.shape)
        q_step = q_step.reshape(self._leaf.base_leaf.mean_param.shape)
        new_log_std = np.log(np.sqrt(1 / Q_step))
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

        self._leaf.base_leaf.std_param = nn.Parameter(th.as_tensor(new_log_std, dtype=th.float,
                                                                   device=self.device))
        self._leaf.base_leaf.mean_param = nn.Parameter(th.as_tensor(new_mean, dtype=th.float,
                                                                    device=self.device))
        return res.x

    def layer_entropy_approx(
            self, layer_index=0, child_entropies: th.Tensor = None,
            sample_size=5, return_child_samples=False, grad_thru_resp=False, return_resp=False, verbose=False,
    ) -> Tuple[th.Tensor, Optional[Dict], Optional[Sample], Optional[th.Tensor]]:
        """

        Args:
            layer_index:
            child_entropies: th.Tensor of shape [w, d, oc of child layer == ic of this layer, r]
            sample_size:
            return_child_samples: If True, the samples of the children are returned.
            grad_thru_resp: If False, the approximation of the responsibility is done without grad.
            return_resp: If True, the unweighted approximated responsibilities are returned as well.
            verbose: If True, return dict of entropy metrics

        Returns: Tuple of
            node_entropies: th.Tensor of size [w, d, oc of child layer == ic of this layer, r]
            logging: Dict or None, depending on verbose
            child_samples: th.Tensor or None, depending on return_child_samples
        """
        assert self.config.C == 1, "Only works for C = 1"
        assert child_entropies is not None or layer_index == 0, \
            "When sampling from a layer other than the leaf layer, child entropies must be provided!"
        logging = {}
        ctx = None
        aux_responsibility = None

        if layer_index == 0:
            ctx = self.sample(
                mode='onehot', n=sample_size, layer_index=layer_index, is_mpe=False,
                do_sample_postprocessing=False,
            )
            # ctx.sample [self.config.I, n, w, self.config.F, self.config.R]
            child_ll = self.forward(
                x=ctx.sample.permute(1, 2, 3, 0, 4),
                layer_index=layer_index, x_needs_permutation=False
            )
            # child_ll [n, w, self.config.F // leaf cardinality, self.config.I, self.config.R]
            node_entropies = -child_ll.mean(dim=0)

            if return_child_samples:
                ctx = self.sample_postprocessing(ctx, split_by_scope=True)
                # ctx.sample [self.config.I, s, n, w, self.config.F, self.config.R]
        else:
            layer = self.layer_index_to_obj(layer_index)

            if isinstance(layer, CrossProduct):
                node_entropies = layer(child_entropies.unsqueeze(0)).squeeze(0)
            else:
                with th.set_grad_enabled(grad_thru_resp):
                    aux_responsibility, ctx = self.approx_responsibilities(
                        layer_index=layer_index, sample_size=sample_size, with_grad=grad_thru_resp,
                        return_sample_ctx=return_child_samples,
                    )
                    # aux_responsibility [w, d, ic, oc, r]

                [weighted_ch_ents, weighted_aux_resp], weight_entropy = self.weigh_tensors(
                    layer_index=layer_index,
                    tensors=[child_entropies.unsqueeze(3), aux_responsibility],
                    return_weight_ent=True
                )
                node_entropies = weight_entropy + weighted_ch_ents + weighted_aux_resp

                if return_child_samples:
                    ctx = self.sample_postprocessing(ctx, split_by_scope=True)

                if verbose:
                    metrics = {
                        'weight_entropy': weight_entropy.detach(),
                        'weighted_child_ent': weighted_ch_ents.detach(),
                        'weighted_aux_resp': weighted_aux_resp.detach(),
                    }
                    logging[layer_index] = self.log_dict_from_metric(metrics)

        return node_entropies, logging, (ctx if return_child_samples else None), (aux_responsibility if return_resp else None)

    def log_dict_from_metric(self, metrics: Dict, rep_dim=-1, batch_dim=0):
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
                    f"{rep_key}/{key}/std": metric.index_select(rep_dim, rep).std(dim=batch_dim).mean().item(),
                })
        return log_dict

    def vips(self, target_dist_callback, steps, step_callback: Callable = None,
             sample_size=10, verbose: Union[bool, Callable] = False, self_prob_penalty=1.0) \
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
            self_prob_penalty:

        Returns:

        """
        logging = {}
        modes = ['spn_prob_penalty']
        leaf_update_mode = 'spn_prob_penalty'
        assert leaf_update_mode in modes

        for step in range(int(steps)):
            verbose_on = verbose(step) if callable(verbose) else verbose
            leaves_need_updating = target_dist_callback is not None
            samples = None
            etas = None

            node_entropies = None
            for layer_index in range(self.num_layers):
                child_entropies = node_entropies
                node_entropies, layer_log, ctx, aux_responsibility = self.layer_entropy_approx(
                    layer_index=layer_index, child_entropies=child_entropies,
                    sample_size=sample_size, return_resp=True,
                    verbose=verbose_on,
                    return_child_samples=target_dist_callback is not None,
                )

                with th.no_grad():
                    if target_dist_callback is not None:
                        if ctx is not None:
                            samples = ctx.sample
                            samples = th.einsum('...ij -> ...ji', samples)
                            # samples [self.config.I := I, s, n, w, self.config.R := R, self.config.F := F]

                            target_prob = target_dist_callback(samples)
                            # target_prob [I, s, n, w, R, F]
                            # In the toy example we are using right now, the target_dist_callback returns a prob for each
                            # feature of F. In a black-box setting, we would first have to fill up the scope with
                            # different evidence-samples of the SPN.
                            # The target_dist_callback would then give us probs for each scope-completed sample.
                            # The last dimension would then be gone instead of size F, wouldn't it?
                            target_prob = target_prob.nan_to_num().sum(-1)
                            target_prob = target_prob.permute(3, 1, 0, 4, 2)
                            # target_prob [w, s, I, R, n] first dims as leaf params, with sample shape last

                        if layer_index == 0:
                            # The repetition dim is flipped with the feature dimension to make it a batch dimension.
                            # This is because in self.forward(samples) we want to evaluate
                            # the probability of the sample on the entire SPN.
                            # If we leave the repetition dimension last, the SPN will evaluate the probabilities
                            # only in the samples' own repetitions.
                            samples = th.einsum('...ijk -> ...jik', samples)
                            # samples [I, s, n, R, w, F]

                            self_probs = self.forward(samples).squeeze(-1)
                            # self_probs [I, s, n, R, w]

                            samples = samples.nan_to_num().sum(-1)
                            # samples is now the non-split-by-scope sample of the leaf layer
                            # TODO we can't do this anymore when we have to fill up the samples
                            samples = samples.permute(4, 1, 0, 3, 2)
                            # samples [w, s, I, R, n]

                            # target_prob = target_prob - 1.0 * self_probs.cpu()

                            # For the leaves, s is F // leaf-cardinality, so we can use the repeat function of the leaves
                            # TODO Not sure if repeat_by_cardinality does what we intend
                            # TODO Do we need to apply the SPN input permutation here?
                            # target_prob = self._leaf.prod.repeat_by_cardinality(target_prob, feature_dim=-4)

                            if leaf_update_mode == 'spn_prob_penalty':
                                self_probs = self_probs.permute(4, 1, 0, 3, 2)
                                # TODO Do we need to run it thru repeat_by_card too?
                                # self_probs [w, s, I, R, n]
                                sum_onto_target_prob = {'spn_prob_penalty': - self_prob_penalty * self_probs }

                            etas = self.natural_param_leaf_update(
                                samples=samples,
                                eta_guess=etas,
                                verbose=verbose_on,
                                target_prob=target_prob,
                                **sum_onto_target_prob,
                            )
                        else:
                            layer = self.layer_index_to_obj(layer_index)
                            if isinstance(layer, Sum):
                                # target_prob [w, s, ic, R, n] first dims as leaf params, with sample shape last
                                target_prob = target_prob.mean(dim=-1).to(self.device)
                                reward = target_prob.unsqueeze(3) + aux_responsibility + child_entropies.unsqueeze(3)
                                if layer_index == self.max_layer_index:
                                    w, d, ic, oc, r = reward.shape
                                    reward = th.einsum('...ijk -> ...kji', reward)
                                    reward = reward.reshape(w, d, ic * r, oc, 1)
                                new_weights = F.log_softmax(reward, dim=2)
                                # new_weights [w, 1, ic * r, 1, 1]
                                layer.weight_param = nn.Parameter(new_weights)

                        layer_log.update(self.log_dict_from_metric({'target_prob': target_prob.detach()}, rep_dim=-2, batch_dim=-1))

                logging.update({layer_index: layer_log})
            if step_callback is not None:
                step_callback(step)
        return logging

    def vi_entropy_approx(self, sample_size=10, grad_thru_resp: bool = False, verbose=False) \
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
            child_entropies, layer_log, ctx, _ = self.layer_entropy_approx(
                layer_index=layer_index, child_entropies=child_entropies,
                sample_size=sample_size, grad_thru_resp=grad_thru_resp,
                verbose=verbose,
            )
            logging.update({layer_index: layer_log})
        return child_entropies.flatten(), logging

    def old2_vi_entropy_approx(self, sample_size=10, verbose=False, aux_resp_ll_with_grad=False,
                              aux_resp_sample_with_grad=False):
        """
        Approximate the entropy of the root sum node via variational inference,
        as done in the Variational Inference by Policy Search paper.
        Args:
            sample_size: Number of samples to approximate the expected entropy of the responsibility with.
            verbose: Return logging data
            aux_resp_ll_with_grad: When approximating the auxiliary responsibility from log-likelihoods
                of child samples, backpropagate the gradient through the LL calculation.
                This argument will be ignored if this function is called in a th.no_grad() context.
            aux_resp_sample_with_grad: May only be True if aux_resp_ll_with_grad is True too. Backpropagate through
                the sampling of the child nodes as well.
                This argument will be ignored if this function is called in a th.no_grad() context.
        """
        assert not self.config.gmm_leaves, "VI entropy not tested on GMM leaves yet."
        assert self.config.C == 1, "For C > 1, we must calculate starting from self._sampling_root!"
        assert not aux_resp_sample_with_grad or (aux_resp_sample_with_grad and aux_resp_ll_with_grad), \
            "aux_resp_sample_with_grad may only be True if aux_resp_ll_with_grad is True as well."
        root_weights_over_rep = th.empty(1)  # For PyCharm
        log_weights = th.empty(1)
        logging = {}

        node_entropies = None
        for layer_index in range(self.num_layers):

            if layer_index == 0:
                ctx = self.sample(
                    mode='onehot', n=sample_size, layer_index=layer_index, is_mpe=False,
                    do_sample_postprocessing=False
                )

                child_ll = self.forward(x=ctx.sample, layer_index=layer_index, x_needs_permutation=False)
                node_entropies = -child_ll.mean(dim=0)
            else:
                layer = self.layer_index_to_obj(layer_index)
                i = layer_index - 1

                if isinstance(layer, CrossProduct):
                    node_entropies = layer(node_entropies.unsqueeze(0)).squeeze(0)
                else:
                    with th.set_grad_enabled(aux_resp_ll_with_grad and th.is_grad_enabled()):
                        ctx = Sample(n=(sample_size,), is_mpe=True)
                        if aux_resp_sample_with_grad and th.is_grad_enabled():
                            # noinspection PyTypeChecker
                            for child_layer in reversed(self._inner_layers[:i]):
                                ctx = child_layer.sample_onehot_style(ctx)
                            child_sample = self._leaf.sample_onehot_style(ctx)
                        else:
                            with th.no_grad():
                                # noinspection PyTypeChecker
                                for child_layer in reversed(self._inner_layers[:i]):
                                    ctx = child_layer.sample_index_style(ctx)
                                child_sample = self._leaf.sample_index_style(ctx)

                        if child_sample.dim() == 4:
                            # child_sample is missing repetition dim. This happens when R=1.
                            child_sample = child_sample.unsqueeze(-1)
                        # The nr_nodes is the number of input channels (ic) to the
                        # current layer - we sampled all its input channels.
                        ic, n, w, d, r = child_sample.shape

                        # Combine first two dims of child_ll.
                        # child_ll [0,0] -> [0], ..., [0, n-1] -> [n-1], [1, 0] -> [n], ...
                        child_sample = child_sample.view(ic * n, w, d, r)
                        child_ll = self._leaf(child_sample)
                        _, w, d, leaf_oc, r = child_ll.shape
                        child_ll = child_ll.view(ic, n, w, d, leaf_oc, r)

                        # We can average over the sample_size dimension with size 'n' here already.
                        child_ll = child_ll.mean(dim=1)

                        # noinspection PyTypeChecker
                        for child_layer in self._inner_layers[:i]:
                            child_ll = child_layer(child_ll)

                        if i == len(self._inner_layers):
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

                            # first reshape the tensor to get the nodes over which we sampled into
                            # the first dimension.
                            # ll = child_ll.permute(0, 4, 1, 2, 3).reshape(ic * r, w, 1, ic)
                            # ll[0,0,:,:] are the log-likelihoods of the samples from the first node computed among the 'ic'
                            # other nodes within that repetition. But the root sum nodes wants the log-likelihoods of the
                            # other 'ic*(r-1)' nodes w.r.t. to that same sample as well! They are all zero.

                        else:
                            ll = layer(child_ll)

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

                    if not i == len(self._inner_layers):
                        log_weights = layer.weights
                    weights = log_weights.exp()

                    aux_responsibility = log_weights.detach() + child_ll - ll
                    weighted_aux_responsibility = th.sum(weights * aux_responsibility, dim=2)
                    if i == len(self._inner_layers):
                        weights = root_weights_over_rep.exp().sum(dim=2).softmax(dim=-1)
                        weighted_aux_responsibility = th.sum(weights * weighted_aux_responsibility, dim=-1)


                    # NEW
                    layer_log_weights = layer.weights
                    lay_weights = layer_log_weights.exp()

                    weight_entropy = th.sum(-lay_weights * layer_log_weights, dim=2)

                    if layer_index == self.max_layer_index:
                        node_entropies = node_entropies.permute(0, 1, 3, 2).reshape(*node_entropies.shape[:2],
                                                                                    np.prod(node_entropies.shape[2:]),
                                                                                    1)
                    weighted_ch_ents = th.sum(node_entropies.unsqueeze(3) * lay_weights, dim=2)

                    if i == len(self._inner_layers):
                        weight_entropy = weight_entropy.squeeze(-1)
                        weighted_ch_ents = weighted_ch_ents.squeeze(-1)

                    node_entropies = weight_entropy + weighted_ch_ents + weighted_aux_responsibility
                    if verbose:
                        weight_entropy = -(layer.weights.exp() * layer.weights).sum(dim=2)
                        metrics = {
                            'weight_entropy': weight_entropy.detach(),
                            'weighted_child_ent': weighted_ch_ents.detach(),
                            'weighted_aux_resp': weighted_aux_responsibility.detach(),
                        }
                        logging[i] = {}
                        for rep in range(weight_entropy.size(-1)):
                            rep_key = f"rep{rep}"
                            rep = th.as_tensor(rep, device=self.device)
                            for key, metric in metrics.items():
                                logging[i].update({
                                    f"{rep_key}/{key}/min": metric.index_select(-1, rep).min().item(),
                                    f"{rep_key}/{key}/max": metric.index_select(-1, rep).max().item(),
                                    f"{rep_key}/{key}/mean": metric.index_select(-1, rep).mean().item(),
                                    f"{rep_key}/{key}/std": metric.index_select(-1, rep).std(dim=0).mean().item(),
                                })

        return node_entropies.flatten(), logging


    def old_vi_entropy_approx(self, sample_size=10, verbose=False, aux_resp_ll_with_grad=False,
                          aux_resp_sample_with_grad=False):
        """
        Approximate the entropy of the root sum node via variational inference,
        as done in the Variational Inference by Policy Search paper.
        Args:
            sample_size: Number of samples to approximate the expected entropy of the responsibility with.
            verbose: Return logging data
            aux_resp_ll_with_grad: When approximating the auxiliary responsibility from log-likelihoods
                of child samples, backpropagate the gradient through the LL calculation.
                This argument will be ignored if this function is called in a th.no_grad() context.
            aux_resp_sample_with_grad: May only be True if aux_resp_ll_with_grad is True too. Backpropagate through
                the sampling of the child nodes as well.
                This argument will be ignored if this function is called in a th.no_grad() context.
        """
        assert not self.config.gmm_leaves, "VI entropy not tested on GMM leaves yet."
        assert self.config.C == 1, "For C > 1, we must calculate starting from self._sampling_root!"
        assert not aux_resp_sample_with_grad or (aux_resp_sample_with_grad and aux_resp_ll_with_grad), \
            "aux_resp_sample_with_grad may only be True if aux_resp_ll_with_grad is True as well."
        root_weights_over_rep = th.empty(1)  # For PyCharm
        log_weights = th.empty(1)
        logging = {}

        child_ll = self._leaf.sample_onehot_style(Sample(n=sample_size, is_mpe=False))
        child_ll = self._leaf(child_ll)
        child_entropies = -child_ll.mean(dim=0, keepdim=True)

        for i in range(len(self._inner_layers) + 1):
            if i < len(self._inner_layers):
                layer = self._inner_layers[i]
            else:
                layer = self.root

            if isinstance(layer, CrossProduct):
                child_entropies = layer(child_entropies)
            else:
                with th.set_grad_enabled(aux_resp_ll_with_grad and th.is_grad_enabled()):
                    ctx = Sample(n=sample_size, is_mpe=False)
                    if aux_resp_sample_with_grad and th.is_grad_enabled():
                        # noinspection PyTypeChecker
                        for child_layer in reversed(self._inner_layers[:i]):
                            ctx = child_layer.sample_onehot_style(ctx)
                        child_sample = self._leaf.sample_onehot_style(ctx)
                    else:
                        with th.no_grad():
                            # noinspection PyTypeChecker
                            for child_layer in reversed(self._inner_layers[:i]):
                                ctx = child_layer.sample_index_style(ctx)
                            child_sample = self._leaf.sample_index_style(ctx)

                    if child_sample.dim() == 4:
                        # child_sample is missing repetition dim. This happens when R=1.
                        child_sample = child_sample.unsqueeze(-1)
                    # The nr_nodes is the number of input channels (ic) to the
                    # current layer - we sampled all its input channels.
                    ic, n, w, d, r = child_sample.shape

                    # Combine first two dims of child_ll.
                    # child_ll [0,0] -> [0], ..., [0, n-1] -> [n-1], [1, 0] -> [n], ...
                    child_sample = child_sample.view(ic * n, w, d, r)
                    child_ll = self._leaf(child_sample)
                    _, w, d, leaf_oc, r = child_ll.shape
                    child_ll = child_ll.view(ic, n, w, d, leaf_oc, r)

                    # We can average over the sample_size dimension with size 'n' here already.
                    child_ll = child_ll.mean(dim=1)

                    # noinspection PyTypeChecker
                    for child_layer in self._inner_layers[:i]:
                        child_ll = child_layer(child_ll)

                    if i == len(self._inner_layers):
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

                        # first reshape the tensor to get the nodes over which we sampled into
                        # the first dimension.
                        # ll = child_ll.permute(0, 4, 1, 2, 3).reshape(ic * r, w, 1, ic)
                        # ll[0,0,:,:] are the log-likelihoods of the samples from the first node computed among the 'ic'
                        # other nodes within that repetition. But the root sum nodes wants the log-likelihoods of the
                        # other 'ic*(r-1)' nodes w.r.t. to that same sample as well! They are all zero.

                    else:
                        ll = layer(child_ll)

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

                weight_entropy = -(layer.weights.exp() * layer.weights).sum(dim=2)
                if not i == len(self._inner_layers):
                    log_weights = layer.weights
                weights = log_weights.exp()
                child_entropies.squeeze_(0)
                weighted_ch_ents = th.sum(child_entropies.unsqueeze(3) * weights, dim=2)
                aux_responsibility = log_weights.detach() + child_ll - ll
                weighted_aux_responsibility = th.sum(weights * aux_responsibility, dim=2)
                if i == len(self._inner_layers):
                    weight_entropy = weight_entropy.squeeze(-1)
                    weights = root_weights_over_rep.exp().sum(dim=2).softmax(dim=-1)
                    weighted_ch_ents = th.sum(weighted_ch_ents * weights, dim=-1)
                    weighted_aux_responsibility = th.sum(weights * weighted_aux_responsibility, dim=-1)
                child_entropies = weight_entropy + weighted_ch_ents + weighted_aux_responsibility
                child_entropies.unsqueeze_(0)
                if verbose:
                    weight_entropy = -(layer.weights.exp() * layer.weights).sum(dim=2)
                    metrics = {
                        'weight_entropy': weight_entropy.detach(),
                        'weighted_child_ent': weighted_ch_ents.detach(),
                        'weighted_aux_resp': weighted_aux_responsibility.detach(),
                    }
                    logging[i] = {}
                    for rep in range(weight_entropy.size(-1)):
                        rep_key = f"rep{rep}"
                        rep = th.as_tensor(rep, device=self.device)
                        for key, metric in metrics.items():
                            logging[i].update({
                                f"{rep_key}/{key}/min": metric.index_select(-1, rep).min().item(),
                                f"{rep_key}/{key}/max": metric.index_select(-1, rep).max().item(),
                                f"{rep_key}/{key}/mean": metric.index_select(-1, rep).mean().item(),
                                f"{rep_key}/{key}/std": metric.index_select(-1, rep).std(dim=0).mean().item(),
                            })

        return child_entropies.flatten(), logging

    def consolidate_weights(self):
        """
            This function calculates the weights of the network if it were a hierarchical mixture model,
            that is, without product layers. These weights are needed for calculating the entropy.
        """
        current_weights: th.Tensor = self.root.weights
        assert current_weights.dim() == 5, "This isn't adopted to the 4-dimensional RatSpn weights yet"

        n, d, ic, oc, _ = current_weights.shape
        # root mean weights have shape [n, 1, S^2*R, C, 1]
        # The sampling root weights the repetitions
        assert oc == 1, "Check if the sampling root weights are calculated correctly for C>1."

        s_root_weights = current_weights.softmax(dim=2).view(n, d, ic // self.config.R, oc, self.config.R)
        s_root_weights = s_root_weights.sum(dim=2, keepdim=True)
        self._sampling_root.consolidated_weights = s_root_weights
        # The weights in the root are reshaped to account for the repetitions,
        # so the product layer can make use of them.
        current_weights = current_weights.view(n, d, ic // self.config.R, oc, self.config.R).softmax(dim=2)
        current_sum: Sum = self.root
        for layer in reversed(self._inner_layers):
            if isinstance(layer, CrossProduct):
                current_weights = layer.consolidate_weights(parent_weights=current_weights)
                current_sum.consolidated_weights = current_weights
            else:  # Is a sum layer
                current_sum: Sum = layer
                current_weights = layer.weights.softmax(dim=2)

    def consolidated_vector_forward(self, leaf_vectors: List[th.Tensor], kernel) -> List[th.Tensor]:
        """
            Performs an upward pass on vectors from the leaf layer. Such vectors have a length of 'cardinality'.
            The upward pass calls 'kernel' at each sum node.
            The results are given into a weighted sum, with the weights being the consolidated weights of the Sum.
            At each product layer, the vectors are concatenated, making them twice as long and halving the number
            of features.
        """
        cardinality = self._leaf.cardinality
        out_channels = self._leaf.out_channels
        r = self.config.R
        n = self._leaf.base_leaf.means.shape[0]

        features = self._leaf.out_features
        if np.log2(features) % 1 != 0.0:
            pad = 2 ** np.ceil(np.log2(features)).astype(np.int) - features
            leaf_vectors = [F.pad(g, pad=[0, 0, 0, 0, 0, 0, 0, pad], mode="constant", value=g.mean().item())
                            if g is not None else None
                            for g in leaf_vectors]
            features += pad
        for layer in self._inner_layers:
            if isinstance(layer, Sum):
                leaf_vectors = kernel(leaf_vectors, layer)
                out_channels = layer.out_channels
            else:
                if layer.in_features != features:
                    # Concatenate grad vectors together, as the features now decrease in number
                    leaf_vectors = [g.view(n, layer.in_features, cardinality * 2, out_channels, r)
                                    for g in leaf_vectors]
                    features = layer.in_features
                    cardinality *= 2
        leaf_vectors = kernel(leaf_vectors, self.root)
        leaf_vectors = [g.view(n, 1, -1, self.config.C, self.config.R)
                        for g in leaf_vectors]
        if leaf_vectors[0].size(2) != self.config.F:
            leaf_vectors = [g[:, :, :self.config.F] for g in leaf_vectors]
        for i in range(self.config.R):
            inv_rand_indices = invert_permutation(self.permutation[:, i])
            for g in leaf_vectors:
                g[:, :, :, :, i] = g[:, :, inv_rand_indices, :, i]
        leaf_vectors = kernel(leaf_vectors, self._sampling_root)
        leaf_vectors = [v.sum(-1).squeeze_(1).squeeze_(-1) for v in leaf_vectors]
        # each tensor in leaf_vectors has shape [n, 1, self.config.F, self.config.C, 1]
        return leaf_vectors

    @staticmethod
    def weighted_sum_kernel(child_grads: th.Tensor, layer: Sum):
        weights = layer.consolidated_weights.unsqueeze(2)
        # Weights is of shape [n, d, 1, ic, oc, r]
        # The extra dimension is created so all elements of the gradient vectors are multiplied by the same
        # weight for that feature and output channel.
        return [(g.unsqueeze(4) * weights).sum(dim=3) for g in child_grads]

    @staticmethod
    def moment_kernel(child_moments: List[th.Tensor], layer: Sum):
        assert layer.consolidated_weights is not None, "No consolidated weights are set for this Sum node!"
        weights = layer.consolidated_weights.unsqueeze(2)
        # Weights is of shape [n, d, 1, ic, oc, r]
        # Create an extra dimension for the mean vector so all elements of the mean vector are multiplied by the same
        # weight for that feature and output channel.

        child_mean = child_moments[0]
        # moments have shape [n, d, cardinality, ic, r]
        # Create an extra 'output channels' dimension, as the weights are separate for each output channel.
        child_mean.unsqueeze_(4)
        mean = child_mean * weights
        # mean has shape [n, d, cardinality, ic, oc, r]
        mean = mean.sum(dim=3)
        # mean has shape [n, d, cardinality, oc, r]
        moments = [mean]

        centered_mean = child_var = 0
        if len(child_moments) >= 2:
            child_var = child_moments[1]
            child_var.unsqueeze_(4)
            centered_mean = child_mean - mean.unsqueeze(4)
            var = child_var + centered_mean**2
            var = var * weights
            var = var.sum(dim=3)
            moments += [var]

        if len(child_moments) >= 3:
            child_skew = child_moments[2]
            skew = 3 * centered_mean * child_var + centered_mean ** 3
            if child_skew is not None:
                child_skew.unsqueeze_(4)
                skew = skew + child_skew
            skew = skew * weights
            skew = skew.sum(dim=3)
            moments += [skew]

        # layer.mean, layer.var, layer.skew = mean, var, skew
        return moments

    def compute_moments(self, order=3):
        moments = self._leaf.moments()
        if len(moments) < order:
            moments += [None] * (order - len(moments))
        return self.consolidated_vector_forward(moments, RatSpn.moment_kernel)

    def compute_gradients(self, x: th.Tensor, with_log_prob_x=False, order=3):
        x = self._randomize(x)
        grads: List = self._leaf.gradient(x, order=order)
        if len(grads) < order:
            grads += [None] * (order - len(grads))
        if with_log_prob_x:
            log_p = self._leaf(x, reduction=None)
            grads += [log_p]
        return self.consolidated_vector_forward(grads, RatSpn.weighted_sum_kernel)

    def sum_node_entropies(self, reduction='mean'):
        inner_sum_ent = []
        norm_inner_sum_ent = []
        for i in range(1, len(self._inner_layers)):
            layer = self._inner_layers[i]
            if isinstance(layer, Sum):
                log_sum_weights: th.Tensor = layer.weights
                if log_sum_weights.dim() == 4:
                    # Only in the Cspn case are the weights already log-normalized
                    log_sum_weights: th.Tensor = th.log_softmax(log_sum_weights, dim=2)
                assert self.sum.weights.dim() == 5, "This isn't adopted to the 4-dimensional RatSpn weights yet"
                nr_cat = log_sum_weights.shape[2]
                max_categ_ent = -np.log(1/nr_cat)
                categ_ent = -(log_sum_weights.exp() * log_sum_weights).sum(dim=2)
                norm_categ_ent = categ_ent / max_categ_ent
                if reduction == 'mean':
                    categ_ent = categ_ent.mean()
                    norm_categ_ent = norm_categ_ent.mean()
                inner_sum_ent.append(categ_ent.unsqueeze(0))
                norm_inner_sum_ent.append(norm_categ_ent.unsqueeze(0))
        inner_sum_ent = th.cat(inner_sum_ent, dim=0)
        norm_inner_sum_ent = th.cat(norm_inner_sum_ent, dim=0)
        log_root_weights = self.root.weights
        if log_root_weights.dim() == 4:
            # Only in the Cspn case are the weights already log-normalized
            log_root_weights: th.Tensor = th.log_softmax(log_root_weights, dim=2)
        nr_cat = log_root_weights.shape[2]
        max_categ_ent = -np.log(1 / nr_cat)
        root_categ_ent = -(log_root_weights.exp() * log_root_weights).sum(dim=2)
        norm_root_categ_ent = root_categ_ent / max_categ_ent
        if reduction == 'mean':
            inner_sum_ent = inner_sum_ent.mean()
            norm_inner_sum_ent = norm_inner_sum_ent.mean()
            root_categ_ent = root_categ_ent.mean()
            norm_root_categ_ent = norm_root_categ_ent.mean()

        return inner_sum_ent, norm_inner_sum_ent, root_categ_ent, norm_root_categ_ent

    @property
    def is_ratspn(self):
        return self.config.is_ratspn

    @property
    def means(self):
        return self._leaf.base_leaf.means

    @property
    def stds(self):
        return self._leaf.base_leaf.stds

    def debug__set_root_weights_dirac(self):
        self.debug__set_weights_dirac(self.max_layer_index)

    def debug__set_weights_dirac(self, layer_index):
        layer = self.layer_index_to_obj(layer_index)
        assert isinstance(layer, Sum), "Given layer index is not a Sum layer!"
        weights = layer.weights
        weights[:] = -100.0
        weights[0, 0, 0, 0, 0] = -1e-3
        weights = weights.log_softmax(dim=2)
        if isinstance(layer.weight_param, nn.Parameter):
            weights = nn.Parameter(weights)
            del layer.weight_param
        layer.weight_param = weights

    def debug__set_dist_params(self):
        """
            Set the dist parameters for debugging purposes.
            Std is set to a very low value.
        """
        min_mean = -100.0
        max_mean = 100.0
        means = th.arange(min_mean, max_mean, (max_mean - min_mean) / self.config.I, device=self.device).unsqueeze(-1)
        means = means.expand_as(self._leaf.base_leaf.means)
        log_stds = self._leaf.base_leaf.stds.data.mul(0).add(3e-5).log()
        if isinstance(self._leaf.base_leaf.means, nn.Parameter):
            means = nn.Parameter(means)
            log_stds = nn.Parameter(log_stds)
            del self._leaf.base_leaf.mean_param
            del self._leaf.base_leaf.std_param
        self._leaf.base_leaf.mean_param = means
        self._leaf.base_leaf.std_param = log_stds
