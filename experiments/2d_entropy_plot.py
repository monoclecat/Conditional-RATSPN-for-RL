import os
import csv
import numpy as np
import torch as th
import gif
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from utils import non_existing_folder_name

if __name__ == "__main__":
    matplotlib.use('Agg')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, help='Directory with model files')
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--grid_points', type=int, default=501)
    parser.add_argument('--make_gif', '-gif', action='store_true', help="Create gif of plots")
    parser.add_argument('--make_video', '-video', action='store_true', help="Create video of plots")
    parser.add_argument('--device', '-dev', type=str, default='cpu', help='cuda or cpu')
    parser.add_argument('--vi_ent_samples', type=int, default=5,
                        help="VI entropy approximation with these many samples")
    parser.add_argument('--mc_ent_samples', type=int, default=100,
                        help="Monte Carlo entropy approximation with these many samples")
    args = parser.parse_args()

    config = None
    cwd = os.path.realpath(args.dir)
    dir_name = os.path.split(cwd)[1]
    for filename in os.listdir(cwd):
        filename = os.fsdecode(filename)
        if filename.endswith('.csv'):
            with open(os.path.join(cwd, filename)) as f:
                reader = csv.DictReader(f)
                config = [r for r in reader][0]
    assert config is not None, f"No config file was found in directory {args.dir}."
    grid_points = args.grid_points
    min_x = -float(config['max_abs_mean'])
    max_x = -min_x

    train_mode = None
    if config['huber']:
        train_mode = 'huber'
    elif config['vi']:
        train_mode = 'vi'
    elif config['mc']:
        train_mode = 'mc'
    assert train_mode is not None, "No train mode was specified in the saved csv file!"

    x = th.linspace(min_x, max_x, grid_points)
    grid = th.stack(th.meshgrid((x, x), indexing='ij'), dim=-1)
    grid = grid.reshape(-1, 2)
    grid_tensor = th.as_tensor(grid, device=args.device, dtype=th.float)

    save_path = os.path.join(args.results_dir, non_existing_folder_name(args.results_dir, dir_name))

    frames = []
    fps = 0
    if args.make_gif:
        fps = 5
        duration = args.duration  # seconds
        n_frames = fps * duration
        make_frame_every = int(config['steps']) / n_frames
    else:
        make_frame_every = 10  # 5000

    frame_save_path = os.path.join(args.dir, non_existing_folder_name(args.dir, 'frames'))

    def pass_grid_to_model(grid):
        global model
        grid = grid.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # Add obligatory leaf output channel and repetition dims
        return model(grid).squeeze(-1).squeeze(-1)

    def scale_to_grid(t: np.ndarray):
        return (t - min_x) / (max_x - min_x) * grid_points

    def exp_view(t: np.ndarray):
        return np.exp(t).reshape(grid_points, grid_points).transpose()

    def dist_imshow(handle, probs, apply_exp_view=True, **kwargs):
        # handle is either an axis or plt
        if apply_exp_view:
            probs = exp_view(probs)
        handle.imshow(probs, extent=[min_x, max_x, max_x, min_x], **kwargs)

    @gif.frame
    def gif_frame(*args, **kwargs):
        plot_dist(*args, **kwargs)

    def plot_dist(probs, huber_ent, vi_ent, mc_ent, mpe, step, train_mode: str):
        fig, (ax1) = plt.subplots(1, figsize=(10, 10), dpi=200)
        ax1.imshow(exp_view(probs))
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
        ax1.set_title(f"Huber ent. LB: {huber_ent:.2f} - VI ent. approx.: {vi_ent:.2f} - MC ent. approx.: {mc_ent:.2f}")
        return fig

    def plot_voronoi(probs, huber_ent, vi_ent, mc_ent, mpe, step, train_mode: str):
        fig, (ax1) = plt.subplots(1, figsize=(10, 10), dpi=200)


    step_pattern = re.compile('step([0-9]+)')

    prev_step = 0
    for filename in tqdm(sorted(os.listdir(os.path.join(cwd, 'models'))), desc='Progress'):
        filename = os.fsdecode(filename)
        model = th.load(os.path.join(cwd, 'models', filename), map_location=args.device).eval()
        grid = grid_tensor.clone()
        if False:
            model.log_stds = model.log_stds * 0
            [model.debug__set_weights_uniform(i) for i in model.sum_layer_indices]
            model.debug__set_dist_params(min_mean=min_x + 5, max_mean=max_x - 5)
            # del model._leaf.base_leaf.permutation
            # model._leaf.base_leaf.permutation = th.as_tensor([0, 2, 1, 3]).unsqueeze(1).repeat(1, 3)
        with th.no_grad():
            if model.config.F > 2:
                pad = th.zeros(grid.size(0), model.config.F - 2, device=args.device) * th.nan
                grid = th.cat((grid, pad), 1)
            if False:
                chunks = th.split(grid, 10000, 0)
                probs = [model(t) for t in chunks]
                probs = th.cat(probs, 0)
            else:
                probs = pass_grid_to_model(grid)

            mpe = model.sample(mode='onehot', n=1, layer_index=1, is_mpe=True).sample
            huber_ent, huber_log = model.huber_entropy_lb(verbose=True)
            vi_ent, vi_log = model.vi_entropy_approx_layerwise(sample_size=args.vi_ent_samples, verbose=True)
            mc_ent = model.monte_carlo_ent_approx(sample_size=args.mc_ent_samples)

        step = step_pattern.search(filename)
        assert step is not None, f"The model file name {filename} didn't contain a 'step[0-9]+' sequence!"
        step = step.groups()[0]
        plot_args = {
            'probs': probs.cpu().numpy(), 'mpe': mpe.cpu().numpy(), 'step': step, 'train_mode': train_mode,
            'huber_ent': huber_ent.item(), 'vi_ent': vi_ent.item(), 'mc_ent': mc_ent.item(),
        }
        if args.make_gif:
            frame = gif_frame(**plot_args)
            frames.append(frame)
        if True:
            fig = plot_dist(**plot_args)
            fig.savefig(os.path.join(frame_save_path, f"plot_{dir_name}__step{step}.jpg"))
            plt.close(fig)
        else:
            plot_dist(**plot_args)
            plt.show()

    if args.make_gif:
        gif.save(frames, f"{save_path}.gif", duration=1 / fps, unit='s')
    elif args.make_video:
        os.system(
            f"ffmpeg -f image2 -r {1/fps} -i {frame_save_path}/* -vcodec mpeg4 -y {save_path}.mp4")
