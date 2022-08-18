import pytest
import distributions
import torch as th


def test_permutation():
    in_features = 100
    out_channels = 8
    num_repetitions = 10
    leaf = distributions.RatNormal(
        in_features=in_features, out_channels=out_channels, ratspn=True, num_repetitions=num_repetitions,
    )
    assert leaf.permutation.shape == leaf.inv_permutation.shape
    assert leaf.permutation.size(0) == in_features
    assert leaf.permutation.size(-1) == num_repetitions
    x = th.arange(in_features).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_repetitions)
    perm_x = leaf.apply_permutation(x)
    assert (x == leaf.apply_inverted_permutation(perm_x)).all()
