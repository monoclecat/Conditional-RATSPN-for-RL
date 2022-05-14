import os
import random
import sys
import time
import csv

import imageio
import numpy as np
import skimage
import torch as th
import torchvision
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from distributions import RatNormal
from cspn import CSPN, CspnConfig, print_cspn_params
from rat_spn import RatSpn, RatSpnConfig
from experiments.vips_paper_exps import build_target_likelihood_planar_n_link

from experiments.train_cspn_mnist_gen import CsvLogger
from experiments.train_mnist import count_params

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-dev', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--epochs', '-ep', type=int, default=100)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--results_dir', type=str, default='../../spn_experiments',
                        help='The base directory where the directory containing the results will be saved to.')
    parser.add_argument('--model_path', type=str,
                        help='Path to the pretrained model. If it is given, '
                             'all other SPN config parameters are ignored.')
    parser.add_argument('--exp_name', '-name', type=str, default='cspn_test',
                        help='Experiment name. The results dir will contain it.')
    parser.add_argument('--repetitions', '-R', type=int, default=5, help='Number of parallel CSPNs to learn at once. ')
    parser.add_argument('--cspn_depth', '-D', type=int, default=3, help='Depth of the CSPN.')
    parser.add_argument('--num_dist', '-I', type=int, default=5, help='Number of Gauss dists per pixel.')
    parser.add_argument('--num_sums', '-S', type=int, default=5, help='Number of sums per RV in each sum layer.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout to apply')
    parser.add_argument('--feat_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN feature layers.')
    parser.add_argument('--sum_param_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN sum param layers.')
    parser.add_argument('--dist_param_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN dist param layers.')
    parser.add_argument('--save_interval', type=int, default=50, help='Epoch interval to save model')
    parser.add_argument('--eval_interval', type=int, default=10, help='Epoch interval to evaluate model')
    parser.add_argument('--verbose', '-V', action='store_true', help='Output more debugging information when running.')
    parser.add_argument('--ratspn', action='store_true', help='Use a RATSPN and not a CSPN')
    parser.add_argument('--ent_approx__sample_size', type=int, default=5,
                        help='When approximating entropy, use this sample size. ')
    parser.add_argument('--ent_loss_coef', type=float, default=0.0,
                        help='Factor for entropy loss. Default 0.0. '
                             'If 0.0, no gradients are calculated w.r.t. the entropy.')
    parser.add_argument('--invert', type=float, default=0.0, help='Probability of an MNIST image being inverted.')
    parser.add_argument('--no_eval_at_start', action='store_true', help='Don\'t evaluate model at the beginning')
    parser.add_argument('--learn_by_sampling', action='store_true', help='Learn in sampling mode.')
    parser.add_argument('--learn_by_sampling__sample_size', type=int, default=10,
                        help='When learning by sampling, this arg sets the number of samples generated for each label.')
    parser.add_argument('--no_tanh', action='store_true', help='Don\'t apply tanh squashing to leaves.')
    parser.add_argument('--sigmoid_std', action='store_true', help='Use sigmoid to set std.')
    parser.add_argument('--no_correction_term', action='store_true', help='Don\'t apply tanh correction term to logprob')
    args = parser.parse_args()

    if args.model_path:
        assert os.path.exists(args.model_path), f"The model_path doesn't exist! {args.model_path}"

    results_dir = os.path.join(args.results_dir, f"results_{args.exp_name}")

    for d in [results_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    if args.device == "cpu":
        device = th.device("cpu")
        use_cuda = False
    else:
        device = th.device("cuda:0")
        use_cuda = True
        th.cuda.benchmark = True
    print("Using device:", device)
    batch_size = args.batch_size

    num_dimensions = 10
    conf_likelihood_var = 4e-2 * np.ones(num_dimensions)
    conf_likelihood_var[0] = 1
    cart_likelihood_var = np.array([1e-4, 1e-4])

    [target_lnpdf, prior, prior_chol] = build_target_likelihood_planar_n_link(num_dimensions, conf_likelihood_var,
                                                                              cart_likelihood_var)

    if not args.model_path:
        config = RatSpnConfig()
        config.C = 1
        config.F = num_dimensions
        config.R = args.repetitions
        config.D = args.cspn_depth
        config.I = args.num_dist
        config.S = args.num_sums
        config.dropout = args.dropout
        config.leaf_base_class = RatNormal
        model = RatSpn(config)
        count_params(model)
        model = model.to(device)
    else:
        print(f"Using pretrained model under {args.model_path}")
        model = th.load(args.model_path, map_location=device)
        model.create_one_hot_in_channel_mapping()
        model.set_no_tanh_log_prob_correction()
    model.train()
    print("Config:", model.config)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print(f"Optimizer: {optimizer}")

    lmbda = 1.0
    sample_interval = 1 if args.verbose else args.eval_interval  # number of epochs
    save_interval = 1 if args.verbose else args.save_interval  # number of epochs

    csv_log = os.path.join(results_dir, f"log_{args.exp_name}.csv")
    logger = CsvLogger(csv_log)

    epoch = 0
    for epoch in range(args.epochs):
        if epoch > 20:
            lmbda = 0.5
        t_start = time.time()
        logger.reset(epoch)

        # Inference

        model.normalize_and_bound_params()
        optimizer.zero_grad()

        sample = model.sample(mode='index', n=3)

        # =========
        mse_loss = ll_loss = ent_loss = vi_ent_approx = th.zeros(1).to(device)
        def bookmark():
            pass
        output: th.Tensor = model(x=data)
        ll_loss = -output.mean()
        loss = (1 - lmbda) * ll_loss + lmbda * loss_ce
        vi_ent_approx = model.vi_entropy_approx(sample_size=10).mean()

        loss.backward()
        optimizer.step()
        logger.add_to_avg_keys(
            nll_loss=ll_loss, mse_loss=mse_loss, ent_loss=ent_loss, loss=loss,
            vi_ent_approx=vi_ent_approx,
        )
        if False:
            for lay_nr, lay_dict in batch_ent_log.items():
                for key, val in lay_dict.items():
                    logger.add_to_avg_keys(**{f"sum_layer{lay_nr}/{key}": val})

        # Log stuff
        if args.verbose:
            logger['time'] = time.time()-t_start
            logger['batch'] = batch_index
            print(logger)
            # print(logger, end="\r")

        t_delta = np.around(time.time()-t_start, 2)
        if epoch % save_interval == (save_interval-1):
            print("Saving model ...")
            th.save(model, os.path.join(model_dir, f"epoch-{epoch:03}_{args.exp_name}.pt"))

        if epoch % sample_interval == (sample_interval-1):
            print("Evaluating model ...")
            save_path = os.path.join(sample_dir, f"epoch-{epoch:03}_{args.exp_name}.png")
            evaluate_sampling(model, save_path, device, img_size)
            save_path = os.path.join(sample_dir, f"mpe-epoch-{epoch:03}_{args.exp_name}.png")
            evaluate_sampling(model, save_path, device, img_size, mpe=True)
            logger['mnist_test_ll'] = evaluate_model(model, device, test_loader, "MNIST test")

        logger.average()
        logger['time'] = t_delta
        logger['batch'] = None
        logger.write()
        print(logger)
    print(1)

