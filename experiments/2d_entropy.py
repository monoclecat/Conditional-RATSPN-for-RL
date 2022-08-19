import numpy as np
from scipy.stats import multivariate_normal as normal_pdf
import torch as th
import matplotlib.pyplot as plt
from rat_spn import RatSpn, RatSpnConfig
from distributions import RatNormal
from experiments.train_mnist import count_params
from experiments.mnist_gen_train import time_delta
from torch import optim
import torch.distributions as dist
import gif
import os
import csv
import time
from utils import *
from tqdm import tqdm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, default='test',
                        help='Experiment name. The results dir will contain it.')
    parser.add_argument('--seed', '-s', type=int, nargs='+', required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--device', '-d', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--ent_approx_sample_size', '-samples', type=int, default=5)
    parser.add_argument('--results_dir', type=str, default='../../spn_experiments',
                        help='The base directory where the directory containing the results will be saved to.')
    parser.add_argument('--additional_grad', action='store_true',
                        help="If True, additional gradients are used (different for each method).")
    parser.add_argument('--log_interval', '-log', type=int, default=1000)
    parser.add_argument('--max_abs_mean', type=int, default=50)
    parser.add_argument('--vi', action='store_true', help="Approximate with VI")
    parser.add_argument('--huber', action='store_true', help="Use Huber lower bound")
    parser.add_argument('--montecarlo', action='store_true', help="Approximate entropy with samples of the root")
    parser.add_argument('--layerwise', action='store_true', help="Use layerwise entropy approximation")
    parser.add_argument('--repetitions', '-R', type=int, default=3, help='Number of parallel CSPNs to learn at once. ')
    parser.add_argument('--features', '-F', type=int, default=2, help='Number of features in the leaf layer')
    parser.add_argument('--num_dist', '-I', type=int, default=4, help='Number of Gauss dists per pixel.')
    parser.add_argument('--num_sums', '-S', type=int, default=3, help='Number of sums per RV in each sum layer.')
    args = parser.parse_args()
    assert args.huber + args.montecarlo + args.vi == 1

    for d in [args.results_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    probs = None
    num_dimensions = 2
    num_true_components = 10

    min_x = -args.max_abs_mean
    max_x = -min_x

    for seed in args.seed:
        th.manual_seed(seed)

        load_path = None
        if load_path is None:
            config = RatSpnConfig()
            config.C = 1
            config.F = args.features
            config.R = args.repetitions
            config.D = int(np.log2(config.F))
            config.I = args.num_dist
            config.S = args.num_sums
            config.dropout = 0.0
            config.leaf_base_class = RatNormal
            config.leaf_base_kwargs = {'min_mean': float(min_x+1), 'max_mean': float(max_x-1)}
            model = RatSpn(config).to(args.device)
            count_params(model)
        else:
            model = th.load(load_path, map_location=args.device)
            # Only for high_ent_ratspn.pt
            w = model.layer_index_to_obj(model.max_layer_index).weight_param.data
            w = w.unsqueeze(0)
            model.layer_index_to_obj(model.max_layer_index).weight_param = th.nn.Parameter(w)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        n_steps = args.steps
        if args.huber:
            exp_name = f"entmax_huberLB_{args.exp_name}" \
                       f"_seed{seed}" \
                       f"_{n_steps}steps" \
                       f"_{args.features}feat"
        elif args.montecarlo:
            exp_name = f"entmax_MCapprox_{args.exp_name}" \
                       f"_{args.ent_approx_sample_size}samples" \
                       f"{'_sampledwithgrad' if args.additional_grad else ''}" \
                       f"_seed{seed}" \
                       f"_{n_steps}steps" \
                       f"_{args.features}feat"
        else:
            exp_name = f"entmax_VIapprox_{args.exp_name}" \
                       f"{'_layerwise' if args.layerwise else ''}" \
                       f"_{args.ent_approx_sample_size}samples" \
                       f"{'_gradthruresp' if args.additional_grad else ''}" \
                       f"_seed{seed}" \
                       f"_{n_steps}steps" \
                       f"_{args.features}feat"
        file_name_base = non_existing_folder_name(args.results_dir, exp_name)
        save_path = os.path.join(args.results_dir, file_name_base)
        model_save_path = os.path.join(save_path, "models")
        os.makedirs(model_save_path, exist_ok=False)
        print(f"Running for {n_steps} steps, saving model every {args.log_interval} steps in {model_save_path}.")

        def bookmark():
            pass
        losses = []
        t_start = time.time()
        plot_at = 150

        def verbose_callback(step):
            return False
            # return step == plot_at
            # return True

        # th.set_anomaly_enabled(True)
        ent_args = {
            'sample_size': args.ent_approx_sample_size,
            'grad_thru_resp': args.additional_grad,
            'verbose': True,
        }

        with open(os.path.join(save_path, f"config_{exp_name}.csv"), 'w') as f:
            w = csv.DictWriter(f, vars(args).keys())
            w.writeheader()
            w.writerow(vars(args))

        for step in tqdm(range(int(n_steps)), desc='Progress'):
            if step % args.log_interval == 0:
                th.save(model, os.path.join(model_save_path, f"{file_name_base}_step{step:06d}.pt"))

            if args.vi:
                ent, log = model.vi_entropy_approx_layerwise(**ent_args)
            elif args.huber:
                ent, log = model.huber_entropy_lb(verbose=True)
            elif args.montecarlo:
                ent = model.monte_carlo_ent_approx(
                    sample_size=args.ent_approx_sample_size, sample_with_grad=args.additional_grad,
                )
            else:
                raise Exception("No entropy calculation mode was selected!")
                # ent, log = model.vi_entropy_approx(**ent_args)
            loss = -ent.mean()
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Finished with seed {seed}.")

