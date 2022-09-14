import os
import random
import sys
import time
import csv

import torch
import wandb
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


def sample_each_digit(model, **kwargs):
    if isinstance(model, CSPN):
        l = F.one_hot(th.arange(10), 10).float().to(model.device)
        s = model.sample(condition=l, **kwargs).sample
    else:
        l = th.arange(10, device=model.device)
        if model.config.C > 1:
            s = model.sample(class_index=l, **kwargs).sample
        else:
            s = model.sample(**kwargs).sample
    return s


def evaluate_sampling(model, img_size, wandb_run=None, eval_ll=True, style='index'):
    log_like = None
    samples_per_label = 10
    sample_kwargs = {
        'n': samples_per_label, 'mode': style, 'is_mpe': False,
    }
    samples = sample_each_digit(model, **sample_kwargs)
    samples = th.einsum('o...r -> ...or', samples)
    if eval_ll:
        if isinstance(model, CSPN):
            log_like = model(x=samples.atanh(), condition=None).mean().item()
        else:
            log_like = model(x=samples.atanh()).mean().item()
    sample_kwargs['is_mpe'] = True
    sample_kwargs['n'] = 1
    mpe_samples = sample_each_digit(model, **sample_kwargs)
    mpe_samples = th.einsum('o...r -> ...or', mpe_samples)
    samples = th.vstack((samples, mpe_samples))
    if wandb_run is not None and log_like is not None:
        wandb.log({f"sample_log_likelihood": log_like})
    if model.config.tanh_squash:
        samples.mul_(0.5).add_(0.5)
    samples = samples.view(-1, *img_size[1:])
    # plt.imshow(samples[0].cpu(), cmap='Greys')
    # plt.show()
    return samples, log_like
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


def horizontal_bar_mask():
    h = w = 28
    horizontal_nan_width = 12
    mask = th.zeros((1, h, w), dtype=th.bool)
    border_left_over = (h - horizontal_nan_width) // 2
    mask[0, border_left_over:(h - border_left_over)] = True
    return mask


def sample_root_children(model, style):
    sample_kwargs = {
        'mode': style, 'is_mpe': False, 'n': 5, 'layer_index': model.max_layer_index - 1,
    }
    if False:
        samples = sample_each_digit(model, **sample_kwargs)
        samples = th.einsum('o...r -> ...or', samples)
    sample_kwargs['is_mpe'] = True
    sample_kwargs['n'] = 1
    mpe_samples = sample_each_digit(model, **sample_kwargs)
    mpe_samples = th.einsum('o...r -> ...or', mpe_samples)
    if False:
        samples = th.vstack((samples, mpe_samples))
    else:
        samples = mpe_samples
    samples = th.einsum('nwfor -> nwofr', samples)
    if model.config.tanh_squash:
        samples.mul_(0.5).add_(0.5)
    samples = samples.view(-1, samples.shape[2], 28, 28, samples.shape[-1])
    return samples


def sample_with_evidence(model, loader, style):
    evidences = None
    num_samples_per_evid = 5

    nan_mask = horizontal_bar_mask().to(model.device)
    evid_color_mask = ~nan_mask.repeat(3, 1, 1)
    evid_color_mask[2] = False

    digit = 0
    for image, label in loader:
        index = th.where(label == digit)[0]
        if len(index) == 0:
            continue
        index = index[0]
        matching_digit = image[index].to(model.device)
        assert len(matching_digit) == 1
        assert (label[index] == digit).all()
        if evidences is None:
            evidences = matching_digit
        else:
            evidences = th.vstack((evidences, matching_digit))
        digit += 1
        if len(evidences) >= 10:
            evidences = evidences[:10]
            break
    assert len(evidences) == 10
    evidences = evidences.unsqueeze(1)
    if model.config.tanh_squash:
        evidences.sub_(0.5).mul_(2).atanh_()
    colored = evidences.repeat(1, 3, 1, 1)
    colored[:, [0, 1]] = 0.0
    show_n_tell_tensors = colored.clone()

    evidences[:, nan_mask] = th.nan
    colored = evidences.repeat(1, 3, 1, 1)
    colored[:, evid_color_mask] = 0.0
    show_n_tell_tensors = th.vstack((show_n_tell_tensors, colored.clone()))

    evidences = evidences.flatten(start_dim=1)
    evidences = evidences.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    sample_kwargs = {
        'mode': style, 'is_mpe': False, 'evidence': evidences, 'n': num_samples_per_evid,
    }

    def reshape_and_color(t):
        h = w = 28
        t = t.clone().squeeze(2).squeeze(0).squeeze(-1)
        # sample = sample.view(num_samples_per_evid, 10, h, w)
        t = t.view(-1, 1, h, w)
        t = t.repeat(1, 3, 1, 1)
        t[:, evid_color_mask] = 0.0
        return t
    show_n_tell_tensors = th.vstack((
        show_n_tell_tensors, reshape_and_color(sample_each_digit(model, **sample_kwargs))
    ))

    # MPE samples
    sample_kwargs['is_mpe'] = True
    sample_kwargs['n'] = 1
    show_n_tell_tensors = th.vstack((
        show_n_tell_tensors, reshape_and_color(sample_each_digit(model, **sample_kwargs))
    ))
    return show_n_tell_tensors


def evaluate_model(model, loader, tag):
    """
    Description for method evaluate_model.

    Args:
        model: PyTorch module or a list of modules, one for each image channel
        loader: Data loader.
        tag (str): Tag for information.

    Returns:
        float: Tuple of loss and accuracy.
    """
    model.eval()
    log_like = []
    with th.no_grad():
        for image, label in loader:
            image = image.flatten(start_dim=1).to(model.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            if model.config.tanh_squash:
                image.sub_(0.5).mul_(2).atanh_()
            if isinstance(model, CSPN):
                label = F.one_hot(label, 10).float().to(model.device)
                log_like.append(model(x=image, condition=label).mean().tolist())
            else:
                log_like.append(model(x=image).mean().tolist())
    mean_ll = np.mean(log_like)
    print(f"{tag} set: Average log-likelihood: {mean_ll:.2f}")
    return mean_ll


def plot_samples(x: th.Tensor, path, wandb_run=None, wandb_caption="", wandb_log_key="unknown_sample", ncol=10):
    """
    Plot a single sample with the target and prediction in the title.

    Args:
        x (th.Tensor): Batch of input images. Has to be shape: [N, C, H, W].
    """
    if x.dim() == 3:
        x.unsqueeze_(1)
    # Clip to valid range
    x[x < 0.0] = 0.0
    x[x > 1.0] = 1.0

    tensors = torchvision.utils.make_grid(x, nrow=ncol, padding=1).cpu()
    if wandb_run is not None:
        wandb_img = wandb.Image(tensors, wandb_caption)
        wandb.log({wandb_log_key: wandb_img})
    arr = tensors.permute(1, 2, 0).numpy()
    arr = skimage.img_as_ubyte(arr)
    imageio.imwrite(path, arr)


class CsvLogger(dict):
    def __init__(self, path, name='Model'):
        super().__init__()
        self.name = name
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
        return_str = f"{self.name} Train Epoch: {self['epoch']} took {time_delta(self['time'])}"
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


def mnist_gen_train(
        results_dir: str,
        dataset_dir: str,
        device: str,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        model_path: str,
        run_name: str,
        proj_name: str,
        eval_interval: int,
        save_interval: int,
        no_wandb: bool,
        ratspn: bool,
        RATSPN_R: int,
        RATSPN_D: int, RATSPN_I: int, RATSPN_S: int, RATSPN_dropout: float,
        CSPN_sum_param_layers: list,
        CSPN_dist_param_layers: list,
        CSPN_feat_layers: list,
        CSPN_cond_layers_inner_act: str,
        min_sigma: float,
        no_tanh: bool,
        no_correction_term: bool,
        verbose: bool,
        sample_onehot: bool,
        invert: float,
        no_eval_at_start: bool,
        ent_approx: bool,
        ent_approx__sample_size: int,
        ent_loss_coef: float,
        learn_by_sampling: bool,
        learn_by_sampling__evidence: bool,
        learn_by_sampling__sample_size: int,
):
    """

    Args:
        device:
        epochs:
        learning_rate: Learning rate
        batch_size:
        results_dir: The base directory where the directory containing the results will be saved to.
        dataset_dir: The base directory to provide to the PyTorch Dataloader.
        model_path: Path to the pretrained model. If it is given, all other SPN config parameters are ignored.
        proj_name: Project name
        run_name: Name of this run. "RATSPN" or "CSPN" will be prepended.
        RATSPN_R: Number of repetitions in RatSPN.
        RATSPN_D: Depth of the SPN.
        RATSPN_I: Number of Gauss dists per pixel.
        RATSPN_S: Number of sums per RV in each sum layer.
        RATSPN_dropout: Dropout to apply
        CSPN_feat_layers: List of sizes of the CSPN feature layers.
        CSPN_sum_param_layers: List of sizes of the CSPN sum param layers.
        CSPN_dist_param_layers: List of sizes of the CSPN dist param layers.
        save_interval: Epoch interval to save model
        eval_interval: Epoch interval to evaluate model
        verbose: Output more debugging information when running.
        ratspn: Use a RATSPN and not a CSPN
        ent_approx: Compute entropy
        sample_onehot: When evaluating model, sample onehot style.
        ent_approx__sample_size: When approximating entropy, use this sample size.
        ent_loss_coef: Factor for entropy loss. Default 0.0. If 0.0, no gradients are calculated w.r.t. the entropy.
        invert: Probability of an MNIST image being inverted.
        no_eval_at_start: Don't evaluate model at the beginning
        learn_by_sampling: Learn in sampling mode.
        learn_by_sampling__sample_size: When learning by sampling, this arg sets the number of samples generated for each label.
        no_tanh: Don't apply tanh squashing to leaves.
        no_wandb: Don't log with wandb.
        no_correction_term: Don't apply tanh correction term to logprob
    """
    if CSPN_sum_param_layers is None:
        CSPN_sum_param_layers = []
    if CSPN_dist_param_layers is None:
        CSPN_dist_param_layers = []
    if CSPN_feat_layers is None:
        CSPN_feat_layers = []
    if model_path:
        assert os.path.exists(model_path), f"The model_path doesn't exist! {model_path}"

    results_dir = os.path.join(results_dir, proj_name)
    run_name = f"{'RATSPN' if ratspn else 'CSPN'}_{run_name}"
    folder = non_existing_folder_name(results_dir, run_name)
    results_dir = os.path.join(results_dir, folder)
    model_dir = os.path.join(results_dir, non_existing_folder_name(results_dir, "models"))
    sample_dir = os.path.join(results_dir, non_existing_folder_name(results_dir, "samples"))
    os.makedirs(dataset_dir, exist_ok=True)

    wandb_run = None
    if not no_wandb:
        if verbose:
            os.environ['WANDB_MODE'] = 'offline'
        else:
            os.environ['WANDB_MODE'] = 'online'
        wandb.login(key=os.environ['WANDB_API_KEY'])
        wandb_run = wandb.init(
            dir=results_dir,
            project=proj_name,
            name=folder,
            group=run_name,
        )

    if device == "cpu":
        device = th.device("cpu")
        use_cuda = False
    else:
        device = th.device("cuda:0")
        use_cuda = True
        # th.cuda.benchmark = True
    print("Using device:", device)

    img_size = (1, 28, 28)  # 3 channels
    cond_size = 10

    # Construct Cspn from config
    train_loader, test_loader = get_mnist_loaders(dataset_dir, use_cuda, batch_size=batch_size, device=device,
                                                  img_side_len=img_size[1], invert=invert, debug_mode=verbose)
    print(f"There are {len(train_loader)} batches per epoch")

    if not model_path:
        if ratspn:
            config = RatSpnConfig()
            config.C = 10
        else:
            config = CspnConfig()
            config.F_cond = (cond_size,)
            config.C = 1
            config.feat_layers = CSPN_feat_layers
            config.sum_param_layers = CSPN_sum_param_layers
            config.dist_param_layers = CSPN_dist_param_layers
            if CSPN_cond_layers_inner_act == 'relu':
                config.cond_layers_inner_act = nn.ReLU
            elif CSPN_cond_layers_inner_act == 'leaky_relu':
                config.cond_layers_inner_act = nn.LeakyReLU
            elif CSPN_cond_layers_inner_act == 'softplus':
                config.cond_layers_inner_act = nn.Softplus
            else:
                config.cond_layers_inner_act = nn.Identity
        config.F = int(np.prod(img_size))
        config.R = RATSPN_R
        config.D = RATSPN_D
        config.I = RATSPN_I
        config.S = RATSPN_S
        config.dropout = RATSPN_dropout
        config.leaf_base_class = RatNormal
        config.leaf_base_kwargs = {
            'no_tanh_log_prob_correction': no_correction_term, 'stds_in_lin_space': True,
        }
        if not no_tanh:
            config.tanh_squash = True
            # config.leaf_base_kwargs = {'min_mean': -5.0, 'max_mean': 5.0}
        else:
            config.leaf_base_kwargs = {'min_mean': 0.0, 'max_mean': 1.0}
        config.leaf_base_kwargs['min_sigma'] = min_sigma
        if ratspn:
            model = RatSpn(config)
            count_params(model)
        else:
            model = CSPN(config)
            print_cspn_params(model)
        model = model.to(device)
        if wandb_run is not None:
            # wandb_run.config.update({'SPN_config': config})
            wandb.log(vars(config))
    else:
        print(f"Using pretrained model under {model_path}")
        model = th.load(model_path, map_location=device)
        # model.create_one_hot_in_channel_mapping()
        # model.set_no_tanh_log_prob_correction()
    model.train()
    print("Config:", model.config)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Optimizer: {optimizer}")

    lmbda = 1.0

    csv_log = os.path.join(results_dir, f"log_{run_name}.csv")
    logger = CsvLogger(csv_log)

    def eval_routine(epoch):
        model.eval()
        with torch.no_grad():
            print("Evaluating model ...")
            save_path = os.path.join(sample_dir, f"epoch-{epoch:04}_{run_name}_samples.png")
            samples, log_like = evaluate_sampling(model, img_size, wandb_run=wandb_run,
                                                  style='onehot' if sample_onehot else 'index')
            wandb_caption = f"Samples at epoch {epoch:04}. Avg. LL: {np.mean(log_like):.2f}"
            print(wandb_caption)
            plot_samples(
                samples, save_path, wandb_run=wandb_run,
                wandb_caption=wandb_caption, wandb_log_key='Samples',
            )

            samples_with_evidence = sample_with_evidence(model, train_loader,
                                                         style='onehot' if sample_onehot else 'index')
            save_path = os.path.join(sample_dir, f"epoch-{epoch:04}_{run_name}_sampling_with_evid.png")
            plot_samples(
                x=samples_with_evidence, path=save_path, wandb_run=wandb_run,
                wandb_caption=f"Sampling with evidence at epoch {epoch:04}", wandb_log_key='Evidence samples',
            )

            root_children_samples = sample_root_children(model, style='onehot' if sample_onehot else 'index')
            save_path = os.path.join(sample_dir, f"epoch-{epoch:04}_all_root_children")
            os.makedirs(save_path, exist_ok=True)
            for r in range(root_children_samples.shape[-1]):
                plot_samples(
                    x=root_children_samples[..., r].reshape(-1, 28, 28),
                    path=os.path.join(save_path, f"epoch-{epoch:04}_{run_name}_root_children_in_rep{r}.png"),
                    wandb_run=wandb_run,
                    wandb_caption=f"Sampling root children at epoch {epoch:04}", wandb_log_key='Root children samples',
                    ncol=root_children_samples.shape[1],
                )

            logger.reset(epoch)
            mnist_test_ll = evaluate_model(model, test_loader, "MNIST test")
            logger['mnist_test_ll'] = mnist_test_ll
            if wandb_run is not None:
                wandb.log({'MNIST test LL': mnist_test_ll})

    epoch = 0
    if not no_eval_at_start:
        eval_routine(epoch)
        logger.average()
        logger.write()
    for epoch in range(epochs):
        model.train()
        if epoch > 20:
            lmbda = 0.5
        t_start = time.time()
        logger.reset(epoch)
        for batch_index, (image, label) in enumerate(train_loader):
            log_dict = {}
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
                output: th.Tensor = model(x=data)
                label = label.unsqueeze(0).unsqueeze(-2).unsqueeze(-1)
                if model.config.C > 1:
                    loss_ce = F.binary_cross_entropy_with_logits(output, label)
                ll_loss = -output.mean()
                loss = (1 - lmbda) * ll_loss + lmbda * loss_ce
                log_dict['nll_loss'] = ll_loss
                log_dict['ce_loss'] = loss_ce
                if ent_approx:
                    vi_ent_approx = model.vi_entropy_approx(sample_size=ent_approx__sample_size).mean()
                    log_dict['vi_ent_approx'] = vi_ent_approx
            else:
                if learn_by_sampling or learn_by_sampling__evidence:
                    if learn_by_sampling__evidence:
                        evidence = data.clone()
                        nan_mask = horizontal_bar_mask().to(model.device)
                        nan_mask = nan_mask.flatten().unsqueeze(-1).unsqueeze(-1)
                        nan_mask = nan_mask.expand_as(data)
                        evidence[nan_mask] = th.nan
                    else:
                        evidence = None
                    sample: th.Tensor = model.sample_onehot_style(
                        condition=label, n=learn_by_sampling__sample_size, evidence=evidence,
                    ).sample
                    sample = th.einsum('o...r -> ...or', sample)
                    if model.config.tanh_squash:
                        sample = sample.clamp(-0.99999, 0.99999).atanh()
                    loss = ((data - sample) ** 2).mean()
                    log_dict['mse_loss'] = loss
                else:
                    output: th.Tensor = model(x=data, condition=label)
                    loss = -output.mean()
                    log_dict['nll_loss'] = loss
                    if ent_approx:
                        vi_ent_approx, batch_ent_log = model.vi_entropy_approx(
                            sample_size=ent_approx__sample_size, condition=label, verbose=True,
                        )
                        vi_ent_approx = vi_ent_approx.mean()
                        log_dict['vi_ent_approx'] = vi_ent_approx
                        if ent_loss_coef > 0.0:
                            ent_loss = -ent_loss_coef * vi_ent_approx
                            log_dict['ent_loss'] = ent_loss
                            loss = loss + ent_loss
            log_dict['loss'] = loss
            loss.backward()
            optimizer.step()
            logger.add_to_avg_keys(**log_dict)
            if wandb_run is not None:
                wandb.log(log_dict)
            if False:
                for lay_nr, lay_dict in batch_ent_log.items():
                    for key, val in lay_dict.items():
                        logger.add_to_avg_keys(**{f"sum_layer{lay_nr}/{key}": val})

            # Log stuff
            if verbose:
                logger['time'] = time.time()-t_start
                logger['batch'] = batch_index
                print(logger)
                # print(logger, end="\r")

        t_delta = np.around(time.time()-t_start, 2)
        if epoch % save_interval == 0 and epoch > 0:
            print("Saving model ...")
            model.save(os.path.join(model_dir, f"epoch-{epoch:04}_{run_name}.pt"))

        if epoch % eval_interval == 0 and epoch > 0:
            eval_routine(epoch)

        logger.average()
        logger['time'] = t_delta
        logger['batch'] = None
        logger.write()
        print(logger)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-dev', type=str, default='cuda', choices=['cpu', 'cuda'])
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
    parser.add_argument('--proj_name', '-proj', type=str, default='test_proj', help='Project name')
    parser.add_argument('--run_name', '-name', type=str, default='test_run',
                        help='Name of this run. "RATSPN" or "CSPN" will be prepended.')
    parser.add_argument('--RATSPN_R', '-R', type=int, default=5, help='Number of repetitions in RatSPN. ')
    parser.add_argument('--RATSPN_D', '-D', type=int, default=3, help='Depth of the SPN.')
    parser.add_argument('--RATSPN_I', '-I', type=int, default=5, help='Number of Gauss dists per pixel.')
    parser.add_argument('--RATSPN_S', '-S', type=int, default=5, help='Number of sums per RV in each sum layer.')
    parser.add_argument('--RATSPN_dropout', type=float, default=0.0, help='Dropout to apply')
    parser.add_argument('--CSPN_feat_layers', '-feat_lay', type=int, nargs='+',
                        help='List of sizes of the CSPN feature layers.')
    parser.add_argument('--CSPN_sum_param_layers', '-sum_lay', type=int, nargs='+',
                        help='List of sizes of the CSPN sum param layers.')
    parser.add_argument('--CSPN_dist_param_layers', '-dist_lay', type=int, nargs='+',
                        help='List of sizes of the CSPN dist param layers.')
    parser.add_argument('--CSPN_cond_layers_inner_act', '-act', type=str,
                        choices=['relu', 'leaky_relu', 'softplus', 'none'],
                        help='Activations of all CSPN layers that provide the parameters.')
    parser.add_argument('--save_interval', '-save', type=int, default=10, help='Epoch interval to save model')
    parser.add_argument('--eval_interval', '-eval', type=int, default=10, help='Epoch interval to evaluate model')
    parser.add_argument('--verbose', '-V', action='store_true', help='Output more debugging information when running.')
    parser.add_argument('--ratspn', action='store_true', help='Use a RATSPN and not a CSPN')
    parser.add_argument('--ent_approx', '-ent', action='store_true', help="Compute entropy")
    parser.add_argument('--sample_onehot', action='store_true', help="When evaluating model, sample onehot style.")
    parser.add_argument('--ent_approx__sample_size', type=int, default=5,
                        help='When approximating entropy, use this sample size. ')
    parser.add_argument('--ent_loss_coef', type=float, default=0.0,
                        help='Factor for entropy loss. Default 0.0. '
                             'If 0.0, no gradients are calculated w.r.t. the entropy.')
    parser.add_argument('--invert', type=float, default=0.0, help='Probability of an MNIST image being inverted.')
    parser.add_argument('--no_eval_at_start', action='store_true', help='Don\'t evaluate model at the beginning')
    parser.add_argument('--learn_by_sampling', action='store_true', help='Learn in sampling mode.')
    parser.add_argument('--learn_by_sampling__evidence', action='store_true',
                        help='Give part of the image as evidence')
    parser.add_argument('--learn_by_sampling__sample_size', type=int, default=10,
                        help='When learning by sampling, this arg sets the number of samples generated for each label.')
    parser.add_argument('--min_sigma', type=float, default=1e-5, help='Minimum standard deviation')
    parser.add_argument('--no_tanh', action='store_true', help='Don\'t apply tanh squashing to leaves.')
    parser.add_argument('--no_wandb', action='store_true', help='Don\'t log with wandb.')
    parser.add_argument('--no_correction_term', action='store_true', help='Don\'t apply tanh correction term to logprob')
    args = parser.parse_args()
    mnist_gen_train(**vars(args))
