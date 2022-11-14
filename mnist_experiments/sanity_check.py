import os
import time

import torch
import wandb
import numpy as np
import torch as th
from torch import optim
import torch.nn as nn

from distributions import RatNormal
from cspn import CSPN, CspnConfig, print_cspn_params
from rat_spn import RatSpn, RatSpnConfig
from mnist_experiments.train_mnist import count_params
from utils import non_existing_folder_name
from mnist_experiments.mnist_gen_train import (
    sample_root_children, evaluate_sampling, horizontal_bar_mask, evaluate_model, plot_samples, CsvLogger
)


def learn_stripes(
        results_dir: str,
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
        RATSPN_D: int, RATSPN_dropout: float,
        CSPN_sum_param_layers: list,
        CSPN_dist_param_layers: list,
        CSPN_feat_layers: list,
        min_sigma: float,
        no_tanh: bool,
        no_correction_term: bool,
        verbose: bool,
        sample_onehot: bool,
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
        model_path: Path to the pretrained model. If it is given, all other SPN config parameters are ignored.
        proj_name: Project name
        run_name: Name of this run. "RATSPN" or "CSPN" will be prepended.
        RATSPN_D: Depth of the SPN.
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
    run_name = f"sanity_check_{'RATSPN' if ratspn else 'CSPN'}_{run_name}"
    folder = non_existing_folder_name(results_dir, run_name)
    results_dir = os.path.join(results_dir, folder)
    model_dir = os.path.join(results_dir, non_existing_folder_name(results_dir, "models"))
    sample_dir = os.path.join(results_dir, non_existing_folder_name(results_dir, "samples"))

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
    if not model_path:
        if ratspn:
            config = RatSpnConfig()
            config.C = 1
        else:
            raise NotImplementedError()
            config = CspnConfig()
            config.F_cond = (cond_size,)
            config.C = 1
            config.feat_layers = CSPN_feat_layers
            config.sum_param_layers = CSPN_sum_param_layers
            config.dist_param_layers = CSPN_dist_param_layers
        config.F = int(np.prod(img_size))
        config.R = 3
        config.D = 3
        config.I = 3
        config.S = 3
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
            wandb_run.config.update({'SPN_config': config})
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

    if ratspn:
        weight_replace = th.ones_like(model.root.weight_param) / model.root.weight_param.shape[2]
        weight_replace = th.log(weight_replace)
        model.root.weight_param = nn.Parameter(weight_replace, requires_grad=False)
    for epoch in range(epochs):
        model.train()
        t_start = time.time()
        logger.reset(epoch)
        low = 0.1
        high = 0.9
        if False:
            image = th.as_tensor([low, high], device=device).repeat(28 // 2)
        else:
            image = th.as_tensor(high, device=device).repeat(28)
        image = image.unsqueeze(-1).repeat(1, 28).flatten()
        image = image.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        assert model.config.I == 3 and model.config.R == 3
        image = image.repeat(batch_size, 1, 1, 3, 3)
        image[..., :, [0, 2]] = low
        image[..., [0, 2], :] = low
        assert th.allclose(image[:, 0, 0, 0, 0], image[0, 0, 0, 0, 0])
        assert th.allclose(image[0, 0, :, 0, 0], image[0, 0, 0, 0, 0])

        if model.config.tanh_squash:
            image.sub_(0.5).mul_(2).atanh_()
        for batch_index in range(100):
            data = image.clone()
            log_dict = {}
            # plt.imshow(image[0].permute(1, 2, 0).cpu(), cmap='Greys')
            # plt.imshow(sample[0, 0].view(*img_size).permute(1, 2, 0).cpu(), cmap='Greys')
            # plt.show()

            # Inference
            optimizer.zero_grad()
            mse_loss = ll_loss = ent_loss = loss_ce = vi_ent_approx = th.zeros(1).to(device)
            def bookmark():
                pass
            if model.is_ratspn:
                if False:
                    output: th.Tensor = model(x=data)
                else:
                    output = model.forward(x=data, layer_index=2)
                ll_loss = -output.mean()
                loss = ll_loss
                log_dict['nll_loss'] = ll_loss
            else:
                raise NotImplementedError()
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
    parser.add_argument('--model_path', type=str,
                        help='Path to the pretrained model. If it is given, '
                             'all other SPN config parameters are ignored.')
    parser.add_argument('--proj_name', '-proj', type=str, default='test_proj', help='Project name')
    parser.add_argument('--run_name', '-name', type=str, default='test_run',
                        help='Name of this run. "RATSPN" or "CSPN" will be prepended.')
    parser.add_argument('--RATSPN_D', '-D', type=int, default=3, help='Depth of the SPN.')
    parser.add_argument('--RATSPN_dropout', type=float, default=0.0, help='Dropout to apply')
    parser.add_argument('--CSPN_feat_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN feature layers.')
    parser.add_argument('--CSPN_sum_param_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN sum param layers.')
    parser.add_argument('--CSPN_dist_param_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN dist param layers.')
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
    learn_stripes(**vars(args))
