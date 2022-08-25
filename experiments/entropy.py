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
import wandb

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_name', '-proj', type=str, default='test_proj', help='Project name')
    parser.add_argument('--run_name', '-name', type=str, default='test_run',
                        help='Name of this run. "RATSPN" or "CSPN" will be prepended.')
    parser.add_argument('--results_dir', type=str, default='../../gmm',
                        help='The base directory where the directory containing the results will be saved to.')
    parser.add_argument('--seed', '-s', type=int, nargs='+', required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--device', '-d', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--vi_sample_size', '-vi_samples', type=int, default=5)
    parser.add_argument('--mc_sample_size', '-mc_samples', type=int, default=50)
    parser.add_argument('--additional_grad', action='store_true',
                        help="If True, additional gradients are used (different for each method).")
    parser.add_argument('--log_interval', '-log', type=int, default=1000)
    parser.add_argument('--max_abs_mean', type=int, default=50)
    parser.add_argument('--objective', '-obj', type=str, help='Entropy objective to maximize.',
                        choices=['vi', 'huber', 'mc'])
    parser.add_argument('--RATSPN_F', '-F', type=int, default=4, help='Number of features in the SPN leaf layer. ')
    parser.add_argument('--RATSPN_R', '-R', type=int, default=3, help='Number of repetitions in RatSPN. ')
    parser.add_argument('--RATSPN_D', '-D', type=int, default=3, help='Depth of the SPN.')
    parser.add_argument('--RATSPN_I', '-I', type=int, default=5, help='Number of Gauss dists per pixel.')
    parser.add_argument('--RATSPN_S', '-S', type=int, default=5, help='Number of sums per RV in each sum layer.')
    parser.add_argument('--min_sigma', type=float, default=1e-5, help='Minimum standard deviation')
    parser.add_argument('--max_sigma', type=float, default=2.0, help='Maximum standard deviation')
    parser.add_argument('--wandb', action='store_true', help='Log with wandb.')
    parser.add_argument('--offline', action='store_true', help='Set wandb to offline mode.')
    parser.add_argument('--stds_sigmoid_bound', action='store_true',
                        help='Bound stds with a sigmoid instead of softplus. ')
    args = parser.parse_args()

    min_x = -args.max_abs_mean
    max_x = -min_x

    for seed in args.seed:
        th.manual_seed(seed)

        load_path = None
        if load_path is None:
            config = RatSpnConfig()
            config.C = 1
            config.F = args.RATSPN_F
            config.R = args.RATSPN_R
            config.D = int(np.log2(config.F))
            config.I = args.RATSPN_I
            config.S = args.RATSPN_S
            config.dropout = 0.0
            config.leaf_base_class = RatNormal
            config.leaf_base_kwargs = {
                'min_mean': float(min_x+1), 'max_mean': float(max_x-1),
                'min_sigma': args.min_sigma, 'max_sigma': args.max_sigma if args.stds_sigmoid_bound else None,
                'stds_in_lin_space': True, 'stds_sigmoid_bound': args.stds_sigmoid_bound,
            }
            model = RatSpn(config).to(args.device)
            count_params(model)
        else:
            model = th.load(load_path, map_location=args.device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        n_steps = args.steps
        if args.objective == 'huber':
            exp_name = f"entmax_huberLB_{args.run_name}" \
                       f"_seed{seed}"
        elif args.objective == 'mc':
            exp_name = f"entmax_MCapprox_{args.run_name}" \
                       f"_{args.mc_sample_size}samples" \
                       f"{'_sampledwithgrad' if args.additional_grad else ''}" \
                       f"_seed{seed}"
        else:
            exp_name = f"entmax_VIapprox_{args.run_name}" \
                       f"_{args.vi_sample_size}samples" \
                       f"{'_gradthruresp' if args.additional_grad else ''}" \
                       f"_seed{seed}"

        args.results_dir = os.path.join(args.results_dir, args.proj_name)
        os.makedirs(args.results_dir, exist_ok=True)
        file_name_base = non_existing_folder_name(args.results_dir, exp_name)
        save_path = os.path.join(args.results_dir, file_name_base)
        model_save_path = os.path.join(save_path, "models")
        os.makedirs(model_save_path, exist_ok=False)
        print(f"Running for {n_steps} steps, saving model every {args.log_interval} steps in {model_save_path}.")

        losses = []
        t_start = time.time()

        # th.set_anomaly_enabled(True)

        wandb_run = None
        if args.wandb:
            if args.offline:
                os.environ['WANDB_MODE'] = 'offline'
            else:
                os.environ['WANDB_MODE'] = 'online'
            wandb.login(key=os.environ['WANDB_API_KEY'])
            wandb_run = wandb.init(
                dir=save_path,
                project=args.proj_name,
                name=file_name_base,
                group=exp_name,
                reinit=True,
                force=True,
                settings=wandb.Settings(start_method="fork"),
            )
            wandb_run.config.update(vars(args))
            wandb_run.config.update({'SPN_config': model.config})

        with open(os.path.join(save_path, f"config_{exp_name}.csv"), 'w') as f:
            if args.wandb:
                args.wandb_run_id = wandb_run.id
            w = csv.DictWriter(f, vars(args).keys())
            w.writeheader()
            w.writerow(vars(args))

        npz_log = {}
        for step in tqdm(range(int(n_steps)+1), desc='Progress'):
            if step % args.log_interval == 0:
                th.save(model, os.path.join(model_save_path, f"{file_name_base}_step{step:06d}.pt"))
                if npz_log != {}:
                    np.savez(os.path.join(save_path, f"metrics_{exp_name}.npz"), npz_log)
            if step == int(n_steps):
                continue

            vi_ent, vi_log = model.vi_entropy_approx_layerwise(
                sample_size=args.vi_sample_size, grad_thru_resp=args.objective == 'vi' and args.additional_grad, verbose=True,
            )
            huber_ent, huber_log = model.huber_entropy_lb(verbose=True)
            mc_ent = model.monte_carlo_ent_approx(
                sample_size=args.mc_sample_size, sample_with_grad=args.objective == 'mc' and args.additional_grad,
            )
            combined_log = {**vi_log, **huber_log}
            combined_log.update({
                'VI_ent_approx': vi_ent.detach().mean().item(),
                'huber_entropy_LB': huber_ent.detach().mean().item(),
                'MC_root_entropy': mc_ent.detach().mean().item(),
            })
            for key, curr_val in combined_log.items():
                curr_val = np.expand_dims(np.asarray(curr_val, dtype='f2'), 0)
                past_vals = npz_log.get(key)
                if past_vals is None:
                    if step > 0:
                        npz_log[key] = np.concatenate((np.zeros((step-1)), curr_val), 0)
                    else:
                        npz_log[key] = curr_val
                else:
                    npz_log[key] = np.concatenate((past_vals, curr_val), 0)

            if args.objective == 'vi':
                loss = -vi_ent.mean()
            elif args.objective == 'huber':
                loss = -huber_ent.mean()
            else:
                loss = -mc_ent.mean()

            if args.wandb:
                wandb.log({
                    **combined_log,
                }, step=step)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Finished with seed {seed}.")
