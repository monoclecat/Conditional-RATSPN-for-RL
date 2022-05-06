#!/usr/bin/env python3
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch as th
from torch import nn


@contextmanager
def provide_evidence(spn: nn.Module, evidence: th.Tensor, requires_grad=False):
    """
    Context manager for sampling with evidence. In this context, the SPN graph is reweighted with the likelihoods
    computed using the given evidence.

    Args:
        spn: SPN that is being used to perform the sampling.
        evidence: Provided evidence. The SPN will perform a forward pass prior to entering this contex.
        requires_grad: If False, runs in th.no_grad() context. (default: False)
    """
    # If no gradients are required, run in no_grad context
    if not requires_grad:
        context = th.no_grad
    else:
        # Else provide null context
        context = nullcontext

    # Run forward pass in given context
    with context():
        # Enter
        for module in spn.modules():
            if hasattr(module, "_enable_input_cache"):
                module._enable_input_cache()

        if evidence is not None:
            _ = spn(evidence)

        # Run in context (nothing needs to be yielded)
        yield

        # Exit
        for module in spn.modules():
            if hasattr(module, "_enable_input_cache"):
                module._disable_input_cache()


@dataclass
class SamplingContext:
    # Number of samples
    n: int = None

    # Number of scopes of the layer we are starting sampling at. This is only needed if the scopes of each
    # sampled nodes should be made explicit in the Spn's full scope.
    scopes: int = None

    # Indices into the out_channels dimension
    parent_indices: th.Tensor = None

    # Indices into the repetition dimension
    repetition_indices: th.Tensor = None

    # MPE flag, if true, will perform most probable explanation sampling
    is_mpe: bool = False

    sample: th.Tensor = None
    sampled_with_evidence: bool = None
    sampling_mode: str = None
    evidence_filled_in: bool = False
    is_split_by_scope: bool = False
    needs_squashing: bool = None

    # If False, the sample is the direct output of the leaves. The leaf features are the scrambled input features.
    # If True, the samples are feature-aligned with the data
    permutation_inverted: bool = False

    def __setattr__(self, key, value):
        if hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"SamplingContext object has no attribute {key}")

    @property
    def is_root(self):
        return self.parent_indices == None and self.repetition_indices == None
