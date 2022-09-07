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
    parser.add_argument('--recursive_sample_size', '-recur_samples', type=int, default=5)
    parser.add_argument('--naive_sample_size', '-naive_samples', type=int, default=50)
    parser.add_argument('--log_interval', '-log', type=int, default=1000)
    parser.add_argument('--max_abs_mean', '-mean', type=int, default=20)
    parser.add_argument('--objective', '-obj', type=str, help='Entropy objective to maximize.',
                        choices=['recursive_aux_no_grad', 'recursive',
                                 'huber', 'huber_hack', 'huber_hack_reverse',
                                 'naive'])
    parser.add_argument('--model_path', '-model', type=str,
                        help='Path to the pretrained model. If it is given, '
                             'all other SPN config parameters are ignored.')
    parser.add_argument('--RATSPN_F', '-F', type=int, default=4, help='Number of features in the SPN leaf layer. ')
    parser.add_argument('--RATSPN_R', '-R', type=int, default=3, help='Number of repetitions in RatSPN. ')
    parser.add_argument('--RATSPN_D', '-D', type=int, default=3, help='Depth of the SPN.')
    parser.add_argument('--RATSPN_I', '-I', type=int, default=5, help='Number of Gauss dists per pixel.')
    parser.add_argument('--RATSPN_S', '-S', type=int, default=5, help='Number of sums per RV in each sum layer.')
    parser.add_argument('--min_sigma', type=float, default=1e-5, help='Minimum standard deviation')
    parser.add_argument('--max_sigma', type=float, default=2.0, help='Maximum standard deviation')
    parser.add_argument('--wandb', action='store_true', help='Log with wandb.')
    parser.add_argument('--offline', action='store_true', help='Set wandb to offline mode.')
    args = parser.parse_args()

    min_x = -args.max_abs_mean
    max_x = -min_x

    args.results_dir = os.path.join(args.results_dir, args.proj_name)
    os.makedirs(args.results_dir, exist_ok=True)

    load_path = args.model_path
    for seed in args.seed:
        th.manual_seed(seed)
        np.random.seed(seed)

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
                'min_sigma': args.min_sigma, 'max_sigma': args.max_sigma,
                'stds_in_lin_space': True, 'stds_sigmoid_bound': True,
            }
            model = RatSpn(config).to(args.device)
            count_params(model)
        else:
            print(f"Using pretrained model {load_path}")
            model = th.load(load_path, map_location=args.device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        n_steps = args.steps
        if args.objective == 'huber':
            exp_name = f"huber_{args.run_name}" \
                       f"_seed{seed}"
        elif args.objective == 'huber_hack':
            exp_name = f"huber_hack_{args.run_name}" \
                       f"_seed{seed}"
        elif args.objective == 'huber_hack_reverse':
            exp_name = f"huber_hack_reverse_{args.run_name}" \
                       f"_seed{seed}"
        elif args.objective == 'naive':
            exp_name = f"naive_{args.run_name}" \
                       f"_{args.naive_sample_size}samples" \
                       f"_seed{seed}"
        elif args.objective == 'recursive':
            exp_name = f"recursive_{args.run_name}" \
                       f"_{args.recursive_sample_size}samples" \
                       f"_seed{seed}"
        elif args.objective == 'recursive_aux_no_grad':
            exp_name = f"recursive_aux_no_grad_{args.run_name}" \
                       f"_{args.recursive_sample_size}samples" \
                       f"_seed{seed}"
        else:
            raise Exception()

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

            recursive_ent, recursive_log = model.recursive_entropy_approx(
                sample_size=args.recursive_sample_size, aux_with_grad=args.objective == 'recursive', verbose=True,
            )
            huber_ent, huber_log = model.huber_entropy_lb(
                verbose=True,
                detach_weights=args.objective == 'huber_hack' or args.objective == 'huber_hack_reverse',
                add_sub_weight_ent=args.objective == 'huber_hack' or args.objective == 'huber_hack_reverse',
                detach_weight_ent_subtraction=args.objective == 'huber_hack_reverse'
            )
            naive_ent = model.naive_entropy_approx(
                sample_size=args.naive_sample_size, sample_with_grad=args.objective == 'naive'
            )
            combined_log = {**recursive_log, **huber_log}
            combined_log.update({
                'recursive_ent_approx': recursive_ent.detach().mean().item(),
                'huber_entropy_LB': huber_ent.detach().mean().item(),
                'naive_root_entropy': naive_ent.detach().mean().item(),
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

            if args.objective == 'recursive' or args.objective == 'recursive_aux_no_grad':
                loss = -recursive_ent.mean()
            elif args.objective == 'huber' or args.objective == 'huber_hack' or args.objective == 'huber_hack_reverse':
                loss = -huber_ent.mean()
            elif args.objective == 'naive':
                loss = -naive_ent.mean()
            else:
                raise Exception()

            if args.wandb:
                wandb.log({
                    **combined_log,
                }, step=step)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Finished with seed {seed}.")

