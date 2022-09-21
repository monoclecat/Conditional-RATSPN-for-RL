import unittest
import os
from experiments.joint_fail_sac_sb3 import joint_failure_sac


class TestJointFailSacSb3(unittest.TestCase):
    def setUp(self) -> None:
        self.exp_base_kwargs = {
            'seed': 10,
            # 'mlp_actor': bool,
            'num_envs': 1,
            'timesteps': 1010,
            'save_interval': 1010,
            'log_interval': 1,
            'env_name': 'HalfCheetah-v3',
            'device': 'cuda',
            'proj_name': 'test_proj',
            'run_name': 'test_run',
            'log_dir': os.environ['TEMP'] + '/joint_fail_test_proj',
            'load_model_path': None,
            'no_wandb': True,
            'no_video': True,
            'ent_coef': 0.1,
            'learning_rate': 3e-4,
            'learning_starts': 1000,
            'buffer_size': 1_000_000,
            'joint_fail_prob': 0.0,
            # 'provide_joint_fail_info_to_actor': bool,
            # 'provide_joint_fail_info_to_critic': bool,
            'repetitions': 3,
            'cspn_depth': 2,
            'num_dist': 3,
            'num_sums': 3,
            'dropout': 0.0,
            'feat_layers': [64],
            'sum_param_layers': [64],
            'dist_param_layers': [64],
            'objective': 'huber',
            'recurs_sample_size': 5,
            'naive_sample_size': 50,
        }

    def test_cspn(self):
        kwargs = self.exp_base_kwargs.copy()
        kwargs['mlp_actor'] = False
        joint_failure_sac(
            **kwargs,
        )


if __name__ == '__main__':
    unittest.main()
