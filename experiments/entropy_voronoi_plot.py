import os
import csv
from typing import Optional

import numpy as np
import torch as th
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from utils import non_existing_folder_name
import wandb
from experiments.entropy_density_plot import get_ents_from_metrics, set_axis_ticks_and_labels


def max_nodes(probs: np.ndarray):
    max_vals = np.amax(probs, axis=(-1, -2), keepdims=True).squeeze(-1)
    probs = probs.reshape(*probs.shape[:2], -1)
    max_nodes_mask = probs == max_vals
    return [node_names[max_nodes_mask[ind]] for ind in np.ndindex(probs.shape[:2])]


def concat_node_names(node_names, enum_thres):
    return [', '.join(i) if len(i) <= enum_thres else f'more than {enum_thres}' for i in node_names]


def plot_dist(probs, huber_ent, vi_ent, mc_ent, mpe, step, train_mode: str, legend_node_enum_thres, cmap, legend_dict):
    labels = concat_node_names(max_nodes(probs), enum_thres=legend_node_enum_thres)
    label_vals = np.asarray([legend_dict[i] for i in labels]).reshape(*probs.shape[:2])

    fig, (ax1) = plt.subplots(1, figsize=(10, 10), dpi=200)
    norm = mpl.colors.Normalize(vmin=0, vmax=len(legend_dict))
    ax_cont = ax1.contourf(label_vals+0.5, norm=norm, cmap=cmap, levels=np.arange(len(legend_dict)+1))

    cbar = fig.colorbar(ax_cont)
    cbar_y = cbar.ax.get_yaxis()
    cbar_y.set_ticks(np.arange(len(legend_dict)) + 0.5, list(legend_dict.keys()))
    cbar_y.labelpad = 15
    cbar.ax.set_ylabel(f"Nodes with max prob", rotation=270)
    if mpe is not None:
        mpe = mpe.reshape(-1, *mpe.shape[-2:])
        mpe = scale_to_grid(mpe)
        for rep in range(mpe.shape[-1]):
            ax1.scatter(mpe[:, 0, rep], mpe[:, 1, rep], s=5, label=f"Modes of rep {rep}")
            # [ax1.text(mpe[i, 0, rep], mpe[i, 1, rep], str(rep), ha="center", va="center") for i in range(mpe.shape[0])]
            [ax1.annotate(str(rep), (mpe[i, 0, rep], mpe[i, 1, rep]), color='white') for i in range(mpe.shape[0])]

        ax1.legend()

    set_axis_ticks_and_labels(ax=ax1, grid_points=probs.shape[0], num_ticks=6,
                              min_x=min_x, max_x=max_x, num_feat=config['RATSPN_F'])

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
    legend_labels = None
    root_children_log_probs = None
    root_children_log_probs__dir_name = 'root_children_log_probs'
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
        elif filename.endswith('.txt') and filename.startswith('legend_labels'):
            with open(os.path.join(cwd, filename)) as f:
                legend_labels = f.readline()
        elif filename.endswith('.npz') and filename.startswith('metrics'):
            metrics = np.load(os.path.join(cwd, filename), allow_pickle=True)
            keys = [i for i in metrics]
            assert len(keys) == 1, f"There should only be one key, but there were {len(keys)}"
            metrics = metrics[keys[0]].tolist()
        elif filename.startswith(root_children_log_probs__dir_name):
            if os.path.isdir(os.path.join(cwd, filename)):  # if we want to save each step separately
                root_children_log_probs = os.listdir(os.path.join(cwd, filename))
            else:
                raise Exception(f"{root_children_log_probs__dir_name} isn't a directory!")

    assert config is not None, f"No config file was found in directory {args.dir}."
    train_mode = config['objective']
    assert train_mode is not None, "No train mode was specified in the saved csv file!"

    def grid_view(t: np.ndarray):
        return t.reshape(grid_points, grid_points).transpose()

    min_x = -float(config['max_abs_mean'])
    max_x = -min_x
    if root_children_log_probs is None:
        os.makedirs(os.path.join(cwd, root_children_log_probs__dir_name))
        steps = None
        probs = None

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
                curr_probs = th.einsum('yxor -> xyor', curr_probs)

            curr_probs = np.asarray(curr_probs, dtype='f4')
            curr_step = step_pattern.search(filename)
            assert curr_step is not None, f"The model file name {filename} didn't contain a 'step[0-9]+' sequence!"
            curr_step = int(curr_step.groups()[0])
            curr_step = np.asarray(curr_step, dtype='i4')
            np.savez(os.path.join(cwd, root_children_log_probs__dir_name, f'root_ch_log_probs__step{curr_step.item():06}'),
                     probs=curr_probs, steps=curr_step)
        root_children_log_probs = os.listdir(os.path.join(cwd, root_children_log_probs__dir_name))

    num_ch = int(config['RATSPN_S']) ** 2
    num_rep = int(config['RATSPN_R'])
    node_names = []
    for o in range(num_ch):
        for r in range(num_rep):
            node_names.append(f"n{o:02}r{r:02}")
    node_names = np.asarray(node_names)

    LABEL_LEN_THRES = 20
    ENUM_THRES = None

    if legend_labels is None:
        label_set = set()
        enum_thres = 2
        for filename in tqdm(sorted(root_children_log_probs), desc="Determining legend and colormap of plots"):
            mpe = None
            probs = np.load(os.path.join(cwd, root_children_log_probs__dir_name, filename))
            probs = probs['probs']
            curr_max_node_names = max_nodes(probs)
            curr_label_set = set(concat_node_names(curr_max_node_names, enum_thres))
            label_set |= curr_label_set
        legend_labels = list(label_set)
        with open(os.path.join(cwd, f'legend_labels_enumthres{enum_thres}.txt'), 'w') as f:
            f.writelines(';'.join(legend_labels))
    else:
        legend_labels = legend_labels.split(';')

    enum_thres = max([i.count(',') for i in legend_labels])+1
    if any(['more' in i for i in legend_labels]):
        legend_labels.sort(key=lambda x: not 'more' in x)
        node_labels = legend_labels[1:]
        overflow_label = [legend_labels[0]]
    else:
        node_labels = legend_labels
        overflow_label = []
    node_labels.sort(key=lambda x: (-len(x), int(x[-2:]), int(x[1:3])))
    legend_labels = overflow_label + node_labels
    legend_indexes = np.arange(len(legend_labels))
    legend_dict = {i: j for i, j in zip(legend_labels, legend_indexes)}

    cmap: Optional[mpl.cm.colors.ListedColormap] = None
    # cmap = mpl.cm.get_cmap('Set3')
    if cmap is not None:
        cmap = mpl.cm.colors.ListedColormap(((0.0, 0.0, 0.0), *cmap.colors))  # black for the 'more than x' label
        label_len_thres = len(cmap.colors)
    else:
        label_len_thres = 15
    if cmap is None:
        cmap = mpl.cm.get_cmap('rainbow', len(legend_dict))

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

    for filename in tqdm(sorted(root_children_log_probs), desc="Reading root_children_log_prob files"):
        mpe = None
        root_children_log_probs = np.load(os.path.join(cwd, root_children_log_probs__dir_name, filename))
        probs = root_children_log_probs['probs']
        steps = root_children_log_probs['steps']

        vi_ent, huber_ent, mc_ent = get_ents_from_metrics(metrics=metrics, step=steps)

        plot_args = {
            'probs': probs, 'mpe': mpe, 'step': steps, 'train_mode': train_mode, 'cmap': cmap,
            'huber_ent': huber_ent, 'vi_ent': vi_ent, 'mc_ent': mc_ent, 'legend_node_enum_thres': enum_thres,
            'legend_dict': legend_dict,
        }
        fig = plot_dist(**plot_args)
        if not os.path.exists(frame_save_path):
            os.makedirs(frame_save_path)
        fig.savefig(os.path.join(frame_save_path, f"plot_{dir_name}__step{steps:06}.jpg"))
        plt.close(fig)
