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


def time_delta(t_delta: float) -> str:
    """
    Convert a timestamp into a human readable timestring.
    Args:
        t_delta (float): Difference between two timestamps of time.time()

    Returns:
        Human readable timestring.
    """
    if t_delta is None:
        return ""
    hours = round(t_delta // 3600)
    minutes = round(t_delta // 60 % 60)
    seconds = round(t_delta % 60)
    return f"{hours}h, {minutes}min, {seconds}s"


def get_mnist_loaders(dataset_dir, use_cuda, device, batch_size, img_side_len, invert=0.0, debug_mode=False):
    """
    Get the MNIST pytorch data loader.

    Args:
        use_cuda: Use cuda flag.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda and not debug_mode else {}

    test_batch_size = batch_size

    # Transform from interval (0.0, 1.0) to (0.01, 0.99) so tanh squash experiments converge better
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize(img_side_len, antialias=False),
                                      transforms.Normalize((-0.010204,), (1.0204,)),
                                      transforms.RandomInvert(p=invert)])
    # Train data loader
    train_loader = th.utils.data.DataLoader(
        datasets.MNIST(dataset_dir, train=True, download=True, transform=transformer),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Test data loader
    test_loader = th.utils.data.DataLoader(
        datasets.MNIST(dataset_dir, train=False, transform=transformer),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )
    return train_loader, test_loader


def evaluate_sampling(model, save_dir, device, img_size, mpe=False, eval_ll=True, style='index'):
    model.eval()
    log_like = []
    label = th.as_tensor(np.arange(10)).to(device)
    samples_per_label = 10
    with th.no_grad():
        if isinstance(model, CSPN):
            label = F.one_hot(label, 10).float().to(device)
            samples = model.sample(n=samples_per_label, mode=style, condition=label, is_mpe=mpe).sample
            samples = th.einsum('o...r -> ...or', samples)
            if eval_ll:
                log_like.append(model(x=samples.atanh(), condition=None).mean().tolist())
        else:
            if model.config.C > 1:
                samples = model.sample(n=samples_per_label, mode=style, class_index=label, is_mpe=mpe).sample
                samples = th.einsum('o...r -> ...or', samples)
            else:
                samples = model.sample(n=samples_per_label, mode=style, is_mpe=mpe).sample
                samples = th.einsum('o...r -> ...or', samples)
            if eval_ll:
                log_like.append(model(x=samples).mean().tolist())
        if model.config.tanh_squash:
            samples.mul_(0.5).add_(0.5)
        samples = samples.view(-1, *img_size[1:])
        # plt.imshow(samples[0].cpu(), cmap='Greys')
        # plt.show()
        plot_samples(samples, save_dir)
        if False and not mpe:
            # To test sampling with evidence
            path_parts = save_dir.split('/')
            base_path = os.path.join('/', *path_parts[:-1], path_parts[-1].split('.')[0])
            for layer_nr in range(model.config.D * 2):
                layer_nr_dir = os.path.join(base_path, f"layer{layer_nr}")
                os.makedirs(layer_nr_dir, exist_ok=True)
                samples = model.sample(mode='index', n=10, layer_index=layer_nr,
                                       post_processing_kwargs={'split_by_scope': True}).sample
                # samples [nr_nodes, scopes, n, w, f, r]
                samples = th.einsum('isnwFR -> RisnwF', samples).unsqueeze(-1)
                # samples [r, nr_nodes, scopes, n, w, f]
                if samples.isnan().any():
                    # evidence_samples = samples.reshape(np.prod(samples.shape[:4]), *samples.shape[4:])
                    evidence_samples = samples
                    if model.config.tanh_squash:
                        evidence_samples = evidence_samples.atanh()
                    if isinstance(model, CSPN):
                        evidence_samples = model.sample(
                            condition=label, mode='index', evidence=evidence_samples, n=(1,), is_mpe=mpe,
                        ).sample
                    else:
                        evidence_samples = model.sample(
                            class_index=label, mode='index', evidence=evidence_samples, n=(1,), is_mpe=mpe,
                        ).sample
                    if model.config.tanh_squash:
                        evidence_samples.mul_(0.5).add_(0.5)
                    evidence_samples = evidence_samples.view(*samples.shape)
                    evidence_samples[~samples.isnan()] = 0.0
                else:
                    evidence_samples = th.zeros_like(samples)
                if model.config.tanh_squash:
                    samples.mul_(0.5).add_(0.5)
                samples[samples.isnan()] = 0.0
                samples = samples.view(*samples.shape[:5], 28, 28, 1)
                evidence_samples = evidence_samples.view(*evidence_samples.shape[:5], 28, 28, 1)
                tmp = th.cat([th.zeros_like(samples), samples, evidence_samples], dim=-1).cpu()
                for rep in range(tmp.shape[0]):
                    rep_dir = os.path.join(layer_nr_dir, f"rep{rep}")
                    os.makedirs(rep_dir, exist_ok=True)
                    for node in range(tmp.shape[1]):
                        node_dir = os.path.join(rep_dir, f"node{node}")
                        os.makedirs(node_dir, exist_ok=True)
                        for scope in range(tmp.shape[2]):
                            feat_dir = os.path.join(node_dir, f"layer{layer_nr}_rep{rep}_node{node}_scope{scope}.png")
                            n = tmp.size(3)
                            w = tmp.size(4)
                            tmp_to_save = tmp[rep, node, scope].view(n*w, 28, 28, 3).permute(0, 3, 1, 2)

                            arr = torchvision.utils.make_grid(tmp_to_save, nrow=10, padding=1).cpu()
                            arr = skimage.img_as_ubyte(arr.permute(1, 2, 0).numpy())
                            imageio.imwrite(feat_dir, arr)

        if False:
            # To test sampling with evidence
            if model.config.tanh_squash:
                samples.sub_(0.5).mul_(2).atanh_()
            zero = samples[0]
            zero[0, :10] = 0.0
            zero[0, 18:] = 0.0
            zero[zero == 0.0] = th.nan
            zero = zero.flatten(start_dim=1).expand(10, -1)
            path_parts = save_dir.split('/')
            evidence_samples = model.sample(condition=label, mode='index', evidence=zero, n=None, is_mpe=mpe).sample
            if model.config.tanh_squash:
                evidence_samples.mul_(0.5).add_(0.5)
            evidence_samples = evidence_samples.view(-1, *img_size[1:])
            plot_samples(evidence_samples, os.path.join('/', *path_parts[:-1], f"{path_parts[-1].split('_')[0]}_slice_evidence_zero.png"))
    result_str = f"{'MPE sample' if mpe else 'Sample'} average log-likelihood: {np.mean(log_like):.2f}"
    print(result_str)


def evaluate_model(model, device, loader, tag):
    """
    Description for method evaluate_model.

    Args:
        model: PyTorch module or a list of modules, one for each image channel
        device: Execution device.
        loader: Data loader.
        tag (str): Tag for information.

    Returns:
        float: Tuple of loss and accuracy.
    """
    model.eval()
    log_like = []
    with th.no_grad():
        for image, label in loader:
            image = image.flatten(start_dim=1).to(device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            if model.config.tanh_squash:
                image.sub_(0.5).mul_(2).atanh_()
            if isinstance(model, CSPN):
                label = F.one_hot(label, 10).float().to(device)
                log_like.append(model(x=image, condition=label).mean().tolist())
            else:
                log_like.append(model(x=image).mean().tolist())
    mean_ll = np.mean(log_like)
    print(f"{tag} set: Average log-likelihood: {mean_ll:.2f}")
    return mean_ll


def plot_samples(x: th.Tensor, path):
    """
    Plot a single sample with the target and prediction in the title.

    Args:
        x (th.Tensor): Batch of input images. Has to be shape: [N, C, H, W].
    """
    x.unsqueeze_(1)
    # Clip to valid range
    x[x < 0.0] = 0.0
    x[x > 1.0] = 1.0

    tensors = torchvision.utils.make_grid(x, nrow=10, padding=1).cpu()
    arr = tensors.permute(1, 2, 0).numpy()
    arr = skimage.img_as_ubyte(arr)
    imageio.imwrite(path, arr)


class CsvLogger(dict):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.other_keys = ['epoch', 'time']
        self.keys_to_avg = [
            'mnist_test_ll', 'nll_loss', 'mse_loss', 'ce_loss', 'ent_loss',
            'vi_ent_approx', 'loss'
        ]
        for i in range(20):
            self.keys_to_avg.append(f"{i}/weight_entropy")
            self.keys_to_avg.append(f"{i}/weighted_child_ent")
            self.keys_to_avg.append(f"{i}/weighted_aux_resp")
        self.no_log_dict = {'batch': None}
        self.reset()
        with open(self.path, 'w') as f:
            w = csv.DictWriter(f, self.keys())
            w.writeheader()

    def add_to_avg_keys(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, th.Tensor):
                v = v.item()
            if k not in self.keys_to_avg:
                raise KeyError(f"{k} not in keys_to_avg")
                self.keys_to_avg += [k]
                self[k] = [v]
            else:
                self[k].append(v)

    def reset(self, epoch: int = None):
        self.update({k: None for k in self.other_keys})
        self.update({k: [] for k in self.keys_to_avg})
        self.no_log_dict.update({k: None for k in self.no_log_dict.keys()})
        if epoch is not None:
            self['epoch'] = epoch

    def average(self):
        self.update({k: np.around(self.mean(k), 2) for k in self.keys_to_avg})

    def write(self):
        with open(self.path, 'a') as f:
            w = csv.DictWriter(f, self.keys())
            w.writerow(self)

    def mean(self, key):
        assert key in self.keys_to_avg, f"key {key} to take mean of is not in keys_to_avg"
        val = self[key]
        if isinstance(val, list):
            if len(val) > 0:
                return np.mean(val)
            else:
                return 0.0
        return val

    def _valid(self, key):
        if key in self.keys() and (mean := self.mean(key)) != 0.0:
            return mean

    def __str__(self):
        return_str = f"Train Epoch: {self['epoch']} took {time_delta(self['time'])}"
        if self.no_log_dict['batch'] is not None:
            return_str += f" @ batch {self.no_log_dict['batch']}"
        if mean := self._valid('nll_loss'):
            return_str += f" - NLL loss: {mean:.2f}"
        if mean := self._valid('mse_loss'):
            return_str += f" - MSE loss: {mean:.4f}"
        if mean := self._valid('ce_loss'):
            return_str += f" - CE loss: {mean:.4f}"
        if mean := self._valid('ent_loss'):
            return_str += f" - Entropy loss: {mean:.2f}"
        if mean := self._valid('mnist_test_ll'):
            return_str += f" - LL orig mnist test set: {mean:.2f}"
        if mean := self._valid('vi_ent_approx'):
            return_str += f" - VI ent. approx.: {mean:.4f}"
        if mean := self._valid('gmm_ent_lb'):
            return_str += f" - GMM ent lower bound: {mean:.4f}"
        if mean := self._valid('gmm_ent_tayl_appr'):
            return_str += f" - GMM ent taylor approx.: {mean:.4f}"
        if mean := self._valid('gmm_H_0'):
            return_str += f" - 1. Taylor: {mean:.4f}"
        if mean := self._valid('gmm_H_2'):
            return_str += f" - 2. Taylor: {mean:.4f}"
        if mean := self._valid('gmm_H_3'):
            return_str += f" - 3. Taylor: {mean:.4f}"
        if mean := self._valid('inner_ent'):
            return_str += f" - Entropy of inner sums: {mean:.4f}|{self.mean('norm_inner_ent'):.2f}%"
        if mean := self._valid('root_ent'):
            return_str += f" - Entropy of root sum: {mean:.4f}|{self.mean('norm_root_ent'):.2f}%"
        for i in range(10):
            if self._valid(f'{i}/weight_entropy') or self._valid(f'{i}/weighted_child_ent') \
                    or self._valid(f'{i}/weighted_aux_resp'):
                return_str += f" - Sum layer {i}: "
                if mean := self._valid(f'{i}/weight_entropy'):
                    return_str += f"Weight ent {mean:.4f} "
                if mean := self._valid(f'{i}/weighted_child_ent'):
                    return_str += f"Weighted child ent {mean:.4f} "
                if mean := self._valid(f'{i}/weighted_aux_resp'):
                    return_str += f"Weighted aux. responsib. {mean:.4f}"

        return return_str

    def __setitem__(self, key, value):
        if isinstance(value, th.Tensor):
            value = value.item()
        if key in self.no_log_dict.keys():
            self.no_log_dict[key] = value
        else:
            super().__setitem__(key, value)


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
    parser.add_argument('--save_interval', '-save', type=int, default=50, help='Epoch interval to save model')
    parser.add_argument('--eval_interval', '-eval', type=int, default=10, help='Epoch interval to evaluate model')
    parser.add_argument('--verbose', '-V', action='store_true', help='Output more debugging information when running.')
    parser.add_argument('--ratspn', action='store_true', help='Use a RATSPN and not a CSPN')
    parser.add_argument('--no_ent_approx', '-no_ent', action='store_true', help="Don't compute entropy")
    parser.add_argument('--sample_onehot', action='store_true', help="When evaluating model, sample onehot style.")
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

    results_dir = os.path.join(args.results_dir, non_existing_folder_name(args.results_dir, f"results_{args.exp_name}"))
    model_dir = os.path.join(results_dir, non_existing_folder_name(results_dir, "models"))
    sample_dir = os.path.join(results_dir, non_existing_folder_name(results_dir, "samples"))
    os.makedirs(args.dataset_dir, exist_ok=True)

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
                                                  img_side_len=img_size[1], invert=args.invert, debug_mode=args.verbose)
    print(f"There are {len(train_loader)} batches per epoch")

    if not args.model_path:
        if args.ratspn:
            config = RatSpnConfig()
            config.C = 1#10
        else:
            config = CspnConfig()
            config.F_cond = (cond_size,)
            config.C = 1
            config.feat_layers = args.feat_layers
            config.sum_param_layers = args.sum_param_layers
            config.dist_param_layers = args.dist_param_layers
        config.F = int(np.prod(img_size))
        config.R = args.repetitions
        config.D = args.cspn_depth
        config.I = args.num_dist
        config.S = args.num_sums
        config.dropout = args.dropout
        config.leaf_base_class = RatNormal
        if not args.no_tanh:
            config.tanh_squash = True
            config.leaf_base_kwargs = {'no_tanh_log_prob_correction': args.no_correction_term}
            # config.leaf_base_kwargs = {'min_mean': -5.0, 'max_mean': 5.0}
        else:
            config.leaf_base_kwargs = {'min_mean': 0.0, 'max_mean': 1.0}
        if args.sigmoid_std:
            config.leaf_base_kwargs['min_sigma'] = 0.1
            config.leaf_base_kwargs['max_sigma'] = 1.0
        if args.ratspn:
            model = RatSpn(config)
            count_params(model)
        else:
            model = CSPN(config)
            print_cspn_params(model)
        model = model.to(device)
    else:
        print(f"Using pretrained model under {args.model_path}")
        model = th.load(args.model_path, map_location=device)
        # model.create_one_hot_in_channel_mapping()
        # model.set_no_tanh_log_prob_correction()
    model.train()
    print("Config:", model.config)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print(f"Optimizer: {optimizer}")

    lmbda = 0.0
    sample_interval = 1 if args.verbose else args.eval_interval  # number of epochs
    save_interval = 1 if args.verbose else args.save_interval  # number of epochs

    csv_log = os.path.join(results_dir, f"log_{args.exp_name}.csv")
    logger = CsvLogger(csv_log)

    epoch = 0
    if not args.no_eval_at_start:
        print("Evaluating model ...")
        save_path = os.path.join(sample_dir, f"epoch-{epoch:03}_{args.exp_name}.png")
        evaluate_sampling(model, save_path, device, img_size,
                          style='onehot' if args.sample_onehot else 'index')
        save_path = os.path.join(sample_dir, f"mpe-epoch-{epoch:03}_{args.exp_name}.png")
        evaluate_sampling(model, save_path, device, img_size, mpe=True,
                          style='onehot' if args.sample_onehot else 'index')
        logger.reset(epoch)
        logger['mnist_test_ll'] = evaluate_model(model, device, test_loader, "MNIST test")
        logger.average()
        logger.write()
    for epoch in range(args.epochs):
        if epoch > 20:
            lmbda = 0.5
        t_start = time.time()
        logger.reset(epoch)
        for batch_index, (image, label) in enumerate(train_loader):
            # Send data to correct device
            label = F.one_hot(label, cond_size).float().to(device)
            image = image.to(device)
            if model.config.tanh_squash:
                image.sub_(0.5).mul_(2).atanh_()
            # plt.imshow(image[0].permute(1, 2, 0).cpu(), cmap='Greys')
            # plt.imshow(sample[0, 0].view(*img_size).permute(1, 2, 0).cpu(), cmap='Greys')
            # plt.show()

            # Inference
            optimizer.zero_grad()
            data = image.reshape(image.shape[0], -1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mse_loss = ll_loss = ent_loss = loss_ce = vi_ent_approx = th.zeros(1).to(device)
            def bookmark():
                pass
            if model.is_ratspn:
                output: th.Tensor = model(x=data).squeeze(1)
                if model.config.C > 1:
                    loss_ce = F.binary_cross_entropy_with_logits(output, label)
                ll_loss = -output.mean()
                loss = (1 - lmbda) * ll_loss + lmbda * loss_ce
                if not args.no_ent_approx:
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
                    if not args.no_ent_approx:
                        vi_ent_approx, batch_ent_log = model.vi_entropy_approx(
                            sample_size=args.ent_approx__sample_size, condition=label, verbose=True,
                        )
                        vi_ent_approx = vi_ent_approx.mean()
                        if args.ent_loss_coef > 0.0:
                            ent_loss = -args.ent_loss_coef * vi_ent_approx
                loss = mse_loss + ll_loss + ent_loss

            loss.backward()
            optimizer.step()
            logger.add_to_avg_keys(
                nll_loss=ll_loss, mse_loss=mse_loss, ce_loss=loss_ce, ent_loss=ent_loss, loss=loss,
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
            model.save(os.path.join(model_dir, f"epoch-{epoch:03}_{args.exp_name}.pt"))

        if epoch % sample_interval == (sample_interval-1):
            print("Evaluating model ...")
            save_path = os.path.join(sample_dir, f"epoch-{epoch:03}_{args.exp_name}.png")
            evaluate_sampling(model, save_path, device, img_size,
                              style='onehot' if args.sample_onehot else 'index')
            save_path = os.path.join(sample_dir, f"mpe-epoch-{epoch:03}_{args.exp_name}.png")
            evaluate_sampling(model, save_path, device, img_size, mpe=True,
                              style='onehot' if args.sample_onehot else 'index')
            logger['mnist_test_ll'] = evaluate_model(model, device, test_loader, "MNIST test")

        logger.average()
        logger['time'] = t_delta
        logger['batch'] = None
        logger.write()
        print(logger)
