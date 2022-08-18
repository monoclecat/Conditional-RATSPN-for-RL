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


if __name__ == '__main__':
    unittest.main()
