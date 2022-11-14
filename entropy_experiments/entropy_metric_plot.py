import os
import csv

import numpy as np
import torch as th
import matplotlib as mpl
import matplotlib.pyplot as plt
import re


if __name__ == "__main__":
    # mpl.use('Agg')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, required=True,
                        help='Directory containing experiment directories (the project directory)')
    args = parser.parse_args()
    device = th.device('cpu')

    assert os.path.isdir(args.dir), f"Path {args.dir} is not a directory!"

    dir_re = ['recursive_aux_no_grad', 'recursive',
              'huber', 'huber_hack',
              'naive']

    child_dirs = os.listdir(args.dir)
    recursive_aux_no_grad_set = set(filter(re.compile('^recursive_aux_no_grad').match, child_dirs))
    recursive_set = set(filter(re.compile('^recursive').match, child_dirs))
    huber_set = set(filter(re.compile('^huber').match, child_dirs))
    huber_hack_set = set(filter(re.compile('^huber_hack').match, child_dirs))
    naive_set = set(filter(re.compile('^naive').match, child_dirs))

    recursive_set -= recursive_aux_no_grad_set
    huber_set -= huber_hack_set

    def color_and_dirs(color, dirs):
        return {'color': color, 'dirs': dirs}

    cmap = mpl.cm.get_cmap('tab10')
    objs_to_plot = {
        'recursive': color_and_dirs(cmap.colors[0], recursive_set),
        'huber': color_and_dirs(cmap.colors[1], huber_set),
        'huber_hack': color_and_dirs(cmap.colors[2], huber_hack_set),
        'naive': color_and_dirs(cmap.colors[3], naive_set),
    }
    assert all(len(i['dirs']) > 0 for i in objs_to_plot.values())
    nr_seeds = [len(i['dirs']) for i in objs_to_plot.values()]
    assert all(i == nr_seeds[0] for i in nr_seeds)

    metrics_operations = {
        'min': np.min,
        'mean': np.mean,
        'median': np.median,
        'max': np.max,
    }

    def re_and_empty_list(regexp: list, op: str, ylim=None, plot_kwargs=None):
        if plot_kwargs is None:
            plot_kwargs = {}
        assert op != ''
        return {
            'regexp': [re.compile(r) for r in regexp],
            'op': op,  # String must contain one of mean, min, max, median
            'ylim': ylim,
            'plot_kwargs': plot_kwargs,
            # A key for every objective to plot will be added
        }

    plots = {
        # plot title: regular expression to match saved metric keys on
        'Root sum node naive entropy approximation':
            re_and_empty_list(['naive_root_entropy'], 'mean', ylim=(None, 120)),
        'Root sum node recursive entropy approximation': re_and_empty_list(['recursive_ent_approx'], 'mean'),
        'Root sum node Huber entropy lower bound': re_and_empty_list(['huber_entropy_LB'], 'mean'),
        'Root sum node weight entropy': re_and_empty_list(['4/rep0/weight_entropy'], 'mean'),
        'Leaf entropies': re_and_empty_list(
            ['0/rep./node_entropy/min', '0/rep./node_entropy/mean', '0/rep./node_entropy/max'],
            'min/median/max',
        ),
        'Layer 2 sum node weight entropies': re_and_empty_list(
            ['2/rep./weight_entropy/min', '2/rep./weight_entropy/mean', '2/rep./weight_entropy/max'],
            'min/median/max',
        ),
    }
    for obj_name in objs_to_plot.keys():
        for plot_title in plots.keys():
            plots[plot_title][obj_name] = {'raw': []}

    for obj_name, obj_dict in objs_to_plot.items():
        nr_seeds = len(obj_dict['dirs'])
        config = None
        metric_keys = None
        for dir_name in obj_dict['dirs']:
            cwd = os.path.join(args.dir, dir_name)
            assert os.path.exists(cwd)
            print(f"Reading from {cwd}")

            metrics = None
            for filename in os.listdir(cwd):
                filename = os.fsdecode(filename)
                if filename.endswith('.csv') and filename.startswith('config'):
                    with open(os.path.join(cwd, filename)) as f:
                        reader = csv.DictReader(f)
                        curr_config = [r for r in reader][0]
                if filename.endswith('.npz') and filename.startswith('metrics'):
                    metrics = np.load(os.path.join(cwd, filename), allow_pickle=True)
                    keys = [i for i in metrics]
                    assert len(keys) == 1, f"There should only be one key, but there were {len(keys)}"
                    metrics = metrics[keys[0]].tolist()
                    if metric_keys is None:
                        metric_keys = list(metrics.keys())
                    else:
                        assert set(metric_keys) == set(metrics.keys())
            assert metrics is not None
            if config is None:
                config = curr_config
                assert config['RATSPN_F'] == '4', "If plots should be generated for an SPN with " \
                                                  "other than 4 leaf features, you need to adapt " \
                                                  "the regular expressions. "
            for plot_contents in plots.values():
                matching_metrics = [set(filter(r.match, metric_keys)) for r in plot_contents['regexp']]
                assert len(matching_metrics[0].union(*matching_metrics[1:])) == np.sum([len(a) for a in matching_metrics])
                matching_metrics = list(matching_metrics[0].union(*matching_metrics[1:]))
                plot_contents[obj_name]['raw'] += [metrics[key] for key in matching_metrics]

        for plot_title, plot_contents in plots.items():
            plot_contents[obj_name]['raw'] = np.vstack(plot_contents[obj_name]['raw'])
            for op in metrics_operations.keys():
                if op in plot_contents['op']:
                    plot_contents[obj_name][op] = metrics_operations[op](plot_contents[obj_name]['raw'], axis=0)

    for plot_title, data_dict in plots.items():
        fig, (ax1) = plt.subplots(1, figsize=(10, 10), dpi=200)
        if data_dict['ylim'] is not None:
            ax1.set_ylim(*data_dict['ylim'])
        for obj_name, obj_dict in objs_to_plot.items():
            obj_data = data_dict[obj_name]
            col = obj_dict['color']
            if 'min' in obj_data.keys() and 'max' in obj_data.keys():
                ax1.fill_between(range(len(obj_data['min'])), obj_data['min'], obj_data['max'],
                                 color=col, alpha=0.5)
            avg = obj_data['mean'] if 'mean' in obj_data.keys() else obj_data['median']
            ax1.plot(avg, color=col, label=obj_name)

            ax1.legend()
        fig.suptitle(plot_title)
        fig.show()
        print(1)
