import os
import csv
import numpy as np
import torch as th
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from utils import non_existing_folder_name
import wandb

if __name__ == "__main__":
    mpl.use('Agg')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, help='Directory with model files', required=True)
    parser.add_argument('--grid_points', type=int, default=500)
    parser.add_argument('--wandb', action='store_true', help='Log with wandb.')
    args = parser.parse_args()
    device = th.device('cpu')

    config = None
    probs = None
    steps = None
    metrics = None
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
        if False and filename.endswith('.npz') and filename.startswith('root_children_log_probs'):
            root_children_log_probs = np.load(os.path.join(cwd, filename))
            probs = root_children_log_probs['probs']
            steps = root_children_log_probs['steps']

    assert config is not None, f"No config file was found in directory {args.dir}."
    train_mode = config['objective']
    assert train_mode is not None, "No train mode was specified in the saved csv file!"

    def grid_view(t: np.ndarray):
        return t.reshape(grid_points, grid_points).transpose()

    min_x = -float(config['max_abs_mean'])
    max_x = -min_x
    if probs is not None:
        print(f"Found saved probs, ignoring args.grid_points.")
        grid_points = probs.shape[-1]
    else:
        grid_points = args.grid_points
        x = th.linspace(min_x, max_x, grid_points)
        grid = th.stack(th.meshgrid((x, x), indexing='ij'), dim=-1)
        grid = grid.reshape(-1, 2)
        grid_tensor = th.as_tensor(grid, device=device, dtype=th.float)

        step_pattern = re.compile('step([0-9]+)')

        for filename in tqdm(sorted(os.listdir(os.path.join(cwd, 'models'))), desc='Computing model probs'):
            filename = os.fsdecode(filename)
            model = th.load(os.path.join(cwd, 'models', filename), map_location=device).eval()
            grid = grid_tensor.clone()
            with th.no_grad():
                if model.config.F > 2:
                    pad = th.zeros(grid.size(0), model.config.F - 2, device=device) * th.nan
                    grid = th.cat((grid, pad), 1)
                grid = grid.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                curr_probs = model.forward(x=grid, layer_index=model.max_layer_index-1).squeeze(1).squeeze(1)
                curr_probs = curr_probs.view(grid_points, grid_points, *curr_probs.shape[1:])
                curr_probs = th.einsum('xyor -> yxor', curr_probs)
            curr_probs = np.expand_dims(np.asarray(curr_probs, dtype='f2'), 0)
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
        np.savez(os.path.join(cwd, 'root_children_log_probs.npz'), probs=probs, steps=steps)
        root_children_log_probs = np.load(os.path.join(cwd, 'root_children_log_probs.npz'))
        probs = root_children_log_probs['probs']
        steps = root_children_log_probs['steps']

    wandb_run = None
    if args.wandb:
        wandb.login(key=os.environ['WANDB_API_KEY'])
        wandb_run = wandb.init(
            resume='must',
            id=config['wandb_run_id'],
            project=config['proj_name'],
            dir=args.dir,
        )

    frame_save_path = f"voronoi_{dir_name}"
    frame_save_path = os.path.join(args.dir, non_existing_folder_name(args.dir, frame_save_path))

    def scale_to_grid(t: np.ndarray):
        return (t - min_x) / (max_x - min_x) * grid_points

    def plot_dist(probs, huber_ent, vi_ent, mc_ent, mpe, step, train_mode: str):
        fig, (ax1) = plt.subplots(1, figsize=(10, 10), dpi=200)
        norm = mpl.colors.Normalize(vmin=args.vmin, vmax=args.vmax, clip=True)
        cmap = mpl.cm.get_cmap('viridis')
        ax1.imshow(grid_view(probs), norm=norm, cmap=cmap)
        num_ticks = 6
        ax1_ticks = np.linspace(0.0, grid_points, num_ticks)
        ax1_ticklabels = np.asarray(np.around(ax1_ticks * ((max_x - min_x) / grid_points) + min_x, decimals=1), dtype='str')
        ax1.set_xticks(ax1_ticks)
        ax1.set_yticks(ax1_ticks)
        ax1.set_xticklabels(ax1_ticklabels)
        ax1.set_yticklabels(ax1_ticklabels)
        ax1.set_xlabel(f"x (feature no. 0 of {config['RATSPN_F']})")
        ax1.set_ylabel(f"y (feature no. 1 of {config['RATSPN_F']})")

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
        elif train_mode == 'vi':
            train_mode = 'VI entropy approximation'
        elif train_mode == 'mc':
            train_mode = 'MC entropy approximation'

        fig.suptitle(f"RatSpn distribution at step {step}, trained with {train_mode}")
        huber_ent_txt = f"Huber ent. LB: {huber_ent:.2f}" if huber_ent is not None else ''
        vi_ent_txt = f"VI ent. approx.: {vi_ent:.2f}" if vi_ent is not None else ''
        mc_ent_txt = f"MC ent. approx.: {mc_ent:.2f}" if mc_ent is not None else ''
        ax1.set_title(' - '.join([huber_ent_txt, vi_ent_txt, mc_ent_txt]))
        return fig

    for i in tqdm(range(len(steps)), desc=f"Creating frames of the {'log ' if args.plot_in_log_space else ''}density"):
        if (num_over := (probs > args.vmax).sum()) > 0:
            print(f"{num_over} probabilities in step {steps[i]} are over {args.vmax}. Max is {probs[i].max():.4f}")
        if False and (num_under := (probs < args.vmin).sum()) > 0:
            print(f"{num_under} probabilities in step {steps[i]:06} are under {args.vmin}. Min is {probs[i].min():.4f}")
        mpe = None

        ents = []
        for m_name in ['VI_ent_approx', 'huber_entropy_LB', 'MC_root_entropy']:
            try:
                ents.append(metrics.get(m_name)[steps[i]])
            except IndexError:
                try:
                    ents.append(metrics.get(m_name)[steps[i]-1])
                except IndexError:
                    ents.append(None)
                    print(f"metric {m_name} didn't exist for step {steps[i]} or {steps[i]-1}")
        vi_ent, huber_ent, mc_ent = ents

        plot_args = {
            'probs': probs[i], 'mpe': mpe, 'step': steps[i], 'train_mode': train_mode,
            'huber_ent': huber_ent, 'vi_ent': vi_ent, 'mc_ent': mc_ent,
        }
        fig = plot_dist(**plot_args)
        if not os.path.exists(frame_save_path):
            os.makedirs(frame_save_path)
        fig.savefig(os.path.join(frame_save_path, f"plot_{dir_name}__step{steps[i]:06}.jpg"))
        plt.close(fig)
