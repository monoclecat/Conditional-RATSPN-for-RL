import os
import csv
import numpy as np
import torch as th
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from utils import non_existing_folder_name


def get_ents_from_metrics(metrics, step):
    ents = []
    for m_name in ['VI_ent_approx', 'huber_entropy_LB', 'MC_root_entropy']:
        try:
            ents.append(metrics.get(m_name)[step])
        except IndexError:
            try:
                ents.append(metrics.get(m_name)[step-1])
            except IndexError:
                ents.append(None)
                print(f"metric {m_name} didn't exist for step {step} or {step - 1}")
    return ents


def set_axis_ticks_and_labels(ax, grid_points, num_ticks, x_abs_max, num_feat):
    ax_ticks = np.linspace(0.0, grid_points, num_ticks)
    ax_ticklabels = np.asarray(np.around(ax_ticks * (2 * x_abs_max / grid_points) - x_abs_max, decimals=1), dtype='str')
    ax.set_xticks(ax_ticks)
    ax.set_yticks(ax_ticks)
    ax.set_xticklabels(ax_ticklabels)
    ax.set_yticklabels(ax_ticklabels)
    ax.set_xlabel(f"x")  # feature no. 0 of {num_feat}
    ax.set_ylabel(f"y")  # feature no. 1 of {num_feat}


if __name__ == "__main__":
    mpl.use('Agg')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, help='Directory with model files', required=True)
    parser.add_argument('--vmin', type=float,
                        help="Maximum probability value on colorbar. If left out will default to the "
                             "max value among all probs during experiment.")
    parser.add_argument('--vmax', type=float,
                        help="Maximum probability value on colorbar. If left out will default to the "
                             "max value among all probs during experiment.")
    parser.add_argument('--x_abs_max', type=int, required=True,
                        help="Maximum absolute range of grid to evaluate probability on")
    parser.add_argument('--plot_in_log_space', '-logspace', action='store_true', help="Plot densities on log scale")
    parser.add_argument('--no_colorbar', action='store_true', help='Log with wandb.')
    args = parser.parse_args()
    device = th.device('cpu')

    config = None
    probs = None
    steps = None
    cwd = os.path.realpath(args.dir)
    print(f"Reading from {cwd}")
    dir_name = os.path.split(cwd)[1]
    for filename in os.listdir(cwd):
        filename = os.fsdecode(filename)
        if filename.endswith('.csv') and filename.startswith('config'):
            with open(os.path.join(cwd, filename)) as f:
                reader = csv.DictReader(f)
                config = [r for r in reader][0]
        if filename.endswith('.npz') and filename.startswith('metrics'):
            metrics = np.load(os.path.join(cwd, filename), allow_pickle=True)
            keys = [i for i in metrics]
            assert len(keys) == 1, f"There should only be one key, but there were {len(keys)}"
            metrics = metrics[keys[0]].tolist()
        if filename.endswith('.npz') and filename.startswith(f'model_log_probs_x_abs_max{args.x_abs_max:03}'):
            model_log_probs = np.load(os.path.join(cwd, filename))
            probs = model_log_probs['probs']
            steps = model_log_probs['steps']

    assert config is not None, f"No config file was found in directory {args.dir}."
    train_mode = config['objective']
    assert train_mode is not None, "No train mode was specified in the saved csv file!"
    assert args.x_abs_max > 0.0

    def grid_view(t: np.ndarray):
        return t.reshape(grid_points, grid_points).transpose()

    min_x = -float(args.x_abs_max)
    max_x = -min_x
    if probs is not None:
        print(f"Found saved probs. Ignoring x_abs_max.")
        grid_points = probs.shape[-1]
    else:
        grid_points = 500
        x = th.linspace(min_x, max_x, grid_points)
        grid = th.stack(th.meshgrid((x, x), indexing='ij'), dim=-1)
        grid = grid.reshape(-1, 2)
        grid_tensor = th.as_tensor(grid, device=device, dtype=th.float)

        def pass_grid_to_model(grid):
            global model
            grid = grid.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # Add obligatory leaf output channel and repetition dims
            return model(grid).squeeze(-1).squeeze(-1)

        step_pattern = re.compile('step([0-9]+)')

        for filename in tqdm(sorted(os.listdir(os.path.join(cwd, 'models'))), desc='Computing model probs'):
            filename = os.fsdecode(filename)
            model = th.load(os.path.join(cwd, 'models', filename), map_location=device).eval()
            setattr(model._leaf.base_leaf, '_means_sigmoid_bound', False)
            grid = grid_tensor.clone()
            if False:
                model.log_stds = model.log_stds * 0
                [model.debug__set_weights_uniform(i) for i in model.sum_layer_indices]
                model.debug__set_dist_params(min_mean=min_x + 5, max_mean=max_x - 5)
                # del model._leaf.base_leaf.permutation
                # model._leaf.base_leaf.permutation = th.as_tensor([0, 2, 1, 3]).unsqueeze(1).repeat(1, 3)
            with th.no_grad():
                if model.config.F > 2:
                    pad = th.zeros(grid.size(0), model.config.F - 2, device=device) * th.nan
                    grid = th.cat((grid, pad), 1)
                if False:
                    chunks = th.split(grid, 10000, 0)
                    curr_probs = [model(t) for t in chunks]
                    curr_probs = th.cat(curr_probs, 0)
                else:
                    curr_probs = pass_grid_to_model(grid)
                curr_probs = curr_probs.cpu().numpy().reshape(grid_points, grid_points).transpose()
            curr_probs = np.expand_dims(np.asarray(curr_probs, dtype='f4'), 0)
            if probs is None:
                probs = curr_probs
            else:
                probs = np.concatenate((probs, curr_probs), 0)

            curr_step = step_pattern.search(filename)
            assert curr_step is not None, f"The model file name {filename} didn't contain a 'step[0-9]+' sequence!"
            curr_step = int(curr_step.groups()[0])
            curr_step = np.expand_dims(np.asarray(curr_step, dtype='i4'), 0)
            if steps is None:
                steps = curr_step
            else:
                steps = np.concatenate((steps, curr_step), 0)
        np.savez(os.path.join(cwd, f'model_log_probs_x_abs_max{args.x_abs_max:03}.npz'), probs=probs, steps=steps)
        model_log_probs = np.load(os.path.join(cwd, f'model_log_probs_x_abs_max{args.x_abs_max:03}.npz'))
        probs = model_log_probs['probs']
        steps = model_log_probs['steps']

    def decimal_ceil(a, precision=0):
        return np.round(a + 0.5 * 10 ** (-precision), precision)
    def decimal_floor(a, precision=0):
        return np.round(a - 0.5 * 10 ** (-precision), precision)
    max_log_prob = probs.max()
    max_prob = np.exp(max_log_prob)
    print(f"The max log prob is {max_log_prob:.4f}, the max linear prob is {max_prob:.4f}")
    if args.vmax is None:
        args.vmax = max_log_prob if args.plot_in_log_space else max_prob
        args.vmax = decimal_ceil(args.vmax, 3)
        print(f"vmax was left empty and has been set to {args.vmax}")
    else:
        print(f"vmax is {args.vmax} ({'log space' if args.plot_in_log_space else 'linear space'}).")
    min_log_prob = probs.min()
    min_prob = np.exp(min_log_prob)
    print(f"The min log prob is {min_log_prob:.4f}, the min linear prob is {min_prob:.4f}")
    if args.vmin is None:
        args.vmin = min_log_prob if args.plot_in_log_space else min_prob
        args.vmin = decimal_floor(args.vmin, 3)
        print(f"vmin was left empty and has been set to {args.vmin}")
    else:
        print(f"vmin is {args.vmin} ({'log space' if args.plot_in_log_space else 'linear space'}).")
    assert args.vmin < args.vmax, "vmin must be lower than vmax!"
    if args.plot_in_log_space:
        assert args.vmin <= 0.0 and args.vmax <= 0.0, "When plotting in log space, vmin and vmax must be <= 0.0!"
        args.vmax = 0.0 if args.vmax > 0.0 else args.vmax
    else:
        args.vmin = abs(0.0 if args.vmin < 0.0 else args.vmin)

    frame_save_path = f"{'log' if args.plot_in_log_space else 'lin'}prob_xabsmax{args.x_abs_max}" \
                      f"_vmin{args.vmin:.4f}" \
                      f"_vmax{args.vmax:.4f}_{'no_colorbar' if args.no_colorbar else ''}_{dir_name}"
    frame_save_path = os.path.join(args.dir, non_existing_folder_name(args.dir, frame_save_path))

    def scale_to_grid(t: np.ndarray):
        return (t - min_x) / (max_x - min_x) * grid_points

    def plot_dist(probs, huber_ent, vi_ent, mc_ent, mpe, step, train_mode: str, no_colorbar: bool, x_abs_max: int):
        fig, (ax1) = plt.subplots(1, figsize=(10, 10), dpi=200)
        norm = mpl.colors.Normalize(vmin=args.vmin, vmax=args.vmax, clip=True)
        cmap = mpl.cm.get_cmap('viridis')
        ax1.imshow(probs.transpose(), norm=norm, cmap=cmap)
        set_axis_ticks_and_labels(ax=ax1, grid_points=probs.shape[-1], num_ticks=6,
                                  x_abs_max=x_abs_max, num_feat=config['RATSPN_F'])

        if not no_colorbar:
            cbar = fig.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1)
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel(f"Probability density [{'log scale' if args.plot_in_log_space else 'linear scale'}]", rotation=270)
        if mpe is not None:
            mpe = mpe.reshape(-1, *mpe.shape[-2:])
            mpe = scale_to_grid(mpe)
            for rep in range(mpe.shape[-1]):
                ax1.scatter(mpe[:, 0, rep], mpe[:, 1, rep], s=5, label=f"Modes of rep {rep}")
                # [ax1.text(mpe[i, 0, rep], mpe[i, 1, rep], str(rep), ha="center", va="center") for i in range(mpe.shape[0])]
                [ax1.annotate(str(rep), (mpe[i, 0, rep], mpe[i, 1, rep]), color='white') for i in range(mpe.shape[0])]

            ax1.legend()
        if train_mode == 'huber':
            train_mode = 'Huber entropy LB'
        elif train_mode == 'huber_hack':
            train_mode = 'augmented Huber entropy LB'
        elif train_mode == 'vi':
            train_mode = 'Recursive entropy approximation'
        elif train_mode == 'mc':
            train_mode = 'Naive entropy approximation'

        ax1.set_title(f"RatSpn distribution at step {step}, trained with {train_mode}")
        if False:
            huber_ent_txt = f"Huber ent. LB: {huber_ent:.2f}" if huber_ent is not None else ''
            vi_ent_txt = f"VI ent. approx.: {vi_ent:.2f}" if vi_ent is not None else ''
            mc_ent_txt = f"MC ent. approx.: {mc_ent:.2f}" if mc_ent is not None else ''
            ax1.set_title(' - '.join([huber_ent_txt, vi_ent_txt, mc_ent_txt]))
        return fig

    if not args.plot_in_log_space:
        probs = np.exp(probs)
    for i in tqdm(range(len(steps)), desc=f"Creating frames of the {'log ' if args.plot_in_log_space else ''}density"):
        if False and (num_over := (probs > args.vmax).sum()) > 0:
            print(f"{num_over} probabilities in step {steps[i]} are over {args.vmax}. Max is {probs[i].max():.4f}")
        if False and (num_under := (probs < args.vmin).sum()) > 0:
            print(f"{num_under} probabilities in step {steps[i]:06} are under {args.vmin}. Min is {probs[i].min():.4f}")
        mpe = None

        vi_ent, huber_ent, mc_ent = get_ents_from_metrics(metrics=metrics, step=steps[i])

        plot_args = {
            'probs': probs[i], 'mpe': mpe, 'step': steps[i], 'train_mode': train_mode,
            'huber_ent': huber_ent, 'vi_ent': vi_ent, 'mc_ent': mc_ent, 'no_colorbar': args.no_colorbar,
            'x_abs_max': args.x_abs_max,
        }
        fig = plot_dist(**plot_args)
        if not os.path.exists(frame_save_path):
            os.makedirs(frame_save_path)
        fig.savefig(os.path.join(frame_save_path, f"plot_{dir_name}__step{steps[i]:06}.jpg"))
        plt.close(fig)
