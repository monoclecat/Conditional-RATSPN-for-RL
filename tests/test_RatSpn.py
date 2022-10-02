import copy
import unittest
from rat_spn import *
from distributions import *


class RatSpnTest(unittest.TestCase):
    def setUp(self) -> None:
        config = RatSpnConfig()
        config.C = 5
        config.F = 64
        config.R = 10
        config.D = 3
        config.I = 3
        config.S = 3
        config.dropout = 0.0
        config.leaf_base_class = RatNormal
        self.model = RatSpn(config)
        self.model.debug__set_dist_params(min_mean=-45.0, max_mean=45.0)

    def test_onehot_and_index_mpe_equal(self):
        onehot_mpe = self.model.sample(mode='onehot', is_mpe=True)
        index_mpe = self.model.sample(mode='index', is_mpe=True)
        assert (onehot_mpe.sample == index_mpe.sample).all()

    def test_log_prob_of_own_samples_when_permuted_and_not_permuted(self):
        ctx = self.model.sample(mode='index', do_sample_postprocessing=False)
        not_inv_permuted = self.model.sample_postprocessing(copy.deepcopy(ctx), invert_permutation=False)
        inv_permuted = self.model.sample_postprocessing(copy.deepcopy(ctx), invert_permutation=True)
        not_inv_permuted.sample = th.einsum('o...r -> ...or', not_inv_permuted.sample.unsqueeze(-1))
        log_prob_not_inv_permuted = self.model.forward(not_inv_permuted.sample, x_needs_permutation=False)
        inv_permuted.sample = th.einsum('o...r -> ...or', inv_permuted.sample)
        log_prob_inv_permuted = self.model.forward(inv_permuted.sample)
        assert (log_prob_not_inv_permuted == log_prob_inv_permuted).all()


if __name__ == '__main__':
    unittest.main()
