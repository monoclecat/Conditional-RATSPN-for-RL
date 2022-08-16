import unittest
from rat_spn import *
from distributions import *


class RatSpnTest(unittest.TestCase):
    def setUp(self) -> None:
        config = RatSpnConfig()
        config.C = 1#10
        config.F = 64
        config.R = 10
        config.D = 3
        config.I = 3
        config.S = 3
        config.dropout = 0.0
        config.leaf_base_class = RatNormal
        self.model = RatSpn(config)
        self.model.debug__set_dist_params(min_mean=-100.0, max_mean=100.0)

    # def test_something(self):
        # sample1 = self.model.sample(mode='index', is_mpe=True, do_sample_postprocessing=True)
        # sample2 = self.model.sample(mode='index', is_mpe=True, do_sample_postprocessing=False, inv_in_leaves=True)
        # assert (sample1.parent_indices == sample2.parent_indices).all()
        # assert (sample1.repetition_indices == sample2.repetition_indices).all()


if __name__ == '__main__':
    unittest.main()
