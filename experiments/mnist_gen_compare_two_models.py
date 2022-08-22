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
from experiments.train_mnist import count_params
from utils import non_existing_folder_name
from mnist_gen_train import *

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-dev', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', '-ep', type=int, default=10000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--results_dir', type=str, default='../../spn_experiments',
                        help='The base directory where the directory containing the results will be saved to.')
    parser.add_argument('--dataset_dir', type=str, default='../../spn_experiments/data',
                        help='The base directory to provide to the PyTorch Dataloader.')
    parser.add_argument('--model1', type=str, help='Path to the first pretrained model.', required=True)
    parser.add_argument('--model2', type=str, help='Path to the first pretrained model.', required=True)
    parser.add_argument('--exp_name', '-name', type=str, default='cspn_test',
                        help='Experiment name. The results dir will contain it.')
    parser.add_argument('--save_interval', '-save', type=int, default=10, help='Epoch interval to save model')
    parser.add_argument('--eval_interval', '-eval', type=int, default=10, help='Epoch interval to evaluate model')
    parser.add_argument('--verbose', '-V', action='store_true', help='Output more debugging information when running.')
    parser.add_argument('--ent_approx', '-ent', action='store_true', help="Compute entropy")
    parser.add_argument('--sample_onehot', action='store_true', help="When evaluating model, sample onehot style.")
    parser.add_argument('--ent_approx__sample_size', type=int, default=5,
                        help='When approximating entropy, use this sample size. ')
    parser.add_argument('--ent_loss_coef', type=float, default=0.0,
                        help='Factor for entropy loss. Default 0.0. '
                             'If 0.0, no gradients are calculated w.r.t. the entropy.')
    parser.add_argument('--no_eval_at_start', action='store_true', help='Don\'t evaluate model at the beginning')
    parser.add_argument('--learn_by_sampling', action='store_true', help='Learn in sampling mode.')
    parser.add_argument('--learn_by_sampling__sample_size', type=int, default=10,
                        help='When learning by sampling, this arg sets the number of samples generated for each label.')
    parser.add_argument('--no_tanh', action='store_true', help='Don\'t apply tanh squashing to leaves.')
    parser.add_argument('--sigmoid_std', action='store_true', help='Use sigmoid to set std.')
    parser.add_argument('--no_correction_term', action='store_true', help='Don\'t apply tanh correction term to logprob')
    args = parser.parse_args()

    assert os.path.exists(args.model1), f"The path model1 doesn't exist! {args.model1}"
    assert os.path.exists(args.model2), f"The path model2 doesn't exist! {args.model2}"

    results_dir = os.path.join(args.results_dir, non_existing_folder_name(args.results_dir, f"results_{args.exp_name}"))
    model_dir = os.path.join(results_dir, non_existing_folder_name(results_dir, "models"))
    sample_dir = os.path.join(results_dir, non_existing_folder_name(results_dir, "samples"))
    os.makedirs(args.dataset_dir, exist_ok=True)

    with open(os.path.join(results_dir, f"args_{args.exp_name}.csv"), 'w') as f:
        w = csv.DictWriter(f, vars(args).keys())
        w.writeheader()
        w.writerow(vars(args))

    if args.device == "cpu":
        device = th.device("cpu")
        use_cuda = False
    else:
        device = th.device("cuda:0")
        use_cuda = True
        # th.cuda.benchmark = True
    print("Using device:", device)
    batch_size = args.batch_size

    img_size = (1, 28, 28)  # 3 channels
    cond_size = 10

    # Construct Cspn from config
    train_loader, test_loader = get_mnist_loaders(args.dataset_dir, use_cuda, batch_size=batch_size, device=device,
                                                  img_side_len=img_size[1], debug_mode=args.verbose)
    print(f"There are {len(train_loader)} batches per epoch")

    model1 = th.load(args.model1, map_location=device)
    model1.train()
    model2 = th.load(args.model2, map_location=device)
    model2.train()
    models = {'model1': model1, 'model2': model2}
    optimizer1 = optim.Adam(model1.parameters(), lr=args.learning_rate)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.learning_rate)
    optimizers = {'model1': optimizer1, 'model2': optimizer2}

    lmbda = 1.0
    sample_interval = 1 if args.verbose else args.eval_interval  # number of epochs
    save_interval = 1 if args.verbose else args.save_interval  # number of epochs

    loggers = {
        'model1': CsvLogger(os.path.join(results_dir, f"model1_log_{args.exp_name}.csv"), name='Model1'),
        'model2': CsvLogger(os.path.join(results_dir, f"model2_log_{args.exp_name}.csv"), name='Model2')
    }

    epoch = 0
    if not args.no_eval_at_start:
        for m in ['model1', 'model2']:
            print(f"Evaluating {m} ...")
            logger = loggers[m]
            save_path = os.path.join(sample_dir, f"{m}_epoch-{epoch:04}_{args.exp_name}.png")
            evaluate_sampling(models[m], save_path, device, img_size,
                              style='onehot' if args.sample_onehot else 'index')
            save_path = os.path.join(sample_dir, f"{m}_mpe-epoch-{epoch:04}_{args.exp_name}.png")
            evaluate_sampling(models[m], save_path, device, img_size, mpe=True,
                              style='onehot' if args.sample_onehot else 'index')
            logger.reset(epoch)
            logger['mnist_test_ll'] = evaluate_model(models[m], device, test_loader, "MNIST test")
            logger.average()
            logger.write()

    for epoch in range(args.epochs):
        if epoch > 20:
            lmbda = 0.5
        t_start = time.time()
        for m in ['model1', 'model2']:
            loggers[m].reset(epoch)
        for batch_index, (image, label) in enumerate(train_loader):
            means_clipped = {'model1': th.zeros_like(model1.means), 'model2': th.zeros_like(model2.means)}
            log_stds_clipped = {'model1': th.zeros_like(model1.means), 'model2': th.zeros_like(model2.means)}
            label = F.one_hot(label, cond_size).float().to(device)
            for m in ['model1', 'model2']:
                data = image.clone().to(device)
                model = models[m]
                logger = loggers[m]
                if model.config.tanh_squash:
                    data.sub_(0.5).mul_(2).atanh_()
                # plt.imshow(data[0].permute(1, 2, 0).cpu(), cmap='Greys')
                # plt.imshow(sample[0, 0].view(*img_size).permute(1, 2, 0).cpu(), cmap='Greys')
                # plt.show()

                # Inference
                optimizers[m].zero_grad()
                data = data.reshape(data.shape[0], -1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                mse_loss = ll_loss = ent_loss = loss_ce = vi_ent_approx = th.zeros(1).to(device)
                def bookmark():
                    pass
                if model.is_ratspn:
                    output: th.Tensor = model(x=data)
                    label = label.unsqueeze(0).unsqueeze(-2).unsqueeze(-1)
                    if model.config.C > 1:
                        loss_ce = F.binary_cross_entropy_with_logits(output, label)
                    ll_loss = -output.mean()
                    loss = (1 - lmbda) * ll_loss + lmbda * loss_ce
                    if args.ent_approx:
                        vi_ent_approx = model.vi_entropy_approx(sample_size=args.ent_approx__sample_size).mean()
                else:
                    if args.learn_by_sampling:
                        sample: th.Tensor = model.sample_onehot_style(condition=label, n=args.learn_by_sampling__sample_size)
                        if model.config.tanh_squash:
                            sample = sample.clamp(-0.99999, 0.99999).atanh()
                        mse_loss: th.Tensor = ((data - sample) ** 2).mean()
                    else:
                        output: th.Tensor = model(x=data, condition=label)
                        ll_loss = -output.mean()
                        if args.ent_approx:
                            vi_ent_approx, batch_ent_log = model.vi_entropy_approx(
                                sample_size=args.ent_approx__sample_size, condition=label, verbose=True,
                            )
                            vi_ent_approx = vi_ent_approx.mean()
                            if args.ent_loss_coef > 0.0:
                                ent_loss = -args.ent_loss_coef * vi_ent_approx
                    loss = mse_loss + ll_loss + ent_loss

                loss.backward()
                optimizers[m].step()
                logger.add_to_avg_keys(
                    nll_loss=ll_loss, mse_loss=mse_loss, ce_loss=loss_ce, ent_loss=ent_loss, loss=loss,
                    vi_ent_approx=vi_ent_approx,
                )

                if args.verbose:
                    logger['time'] = time.time()-t_start
                    logger['batch'] = batch_index
                    print(logger)

        t_delta = np.around(time.time()-t_start, 2)
        if epoch % save_interval == (save_interval-1):
            for m in ['model1', 'model2']:
                print(f"Saving {m} ...")
                models[m].save(os.path.join(model_dir, f"{m}_epoch-{epoch:04}_{args.exp_name}.pt"))

        if epoch % sample_interval == (sample_interval-1):
            for m in ['model1', 'model2']:
                print(f"Evaluating {m} ...")
                save_path = os.path.join(sample_dir, f"{m}_epoch-{epoch:04}_{args.exp_name}.png")
                evaluate_sampling(models[m], save_path, device, img_size,
                                  style='onehot' if args.sample_onehot else 'index')
                save_path = os.path.join(sample_dir, f"{m}_mpe-epoch-{epoch:04}_{args.exp_name}.png")
                evaluate_sampling(models[m], save_path, device, img_size, mpe=True,
                                  style='onehot' if args.sample_onehot else 'index')

    for m in ['model1', 'model2']:
        logger = loggers[m]
        logger.average()
        logger['time'] = t_delta
        logger['batch'] = None
        logger.write()
        print(logger)
