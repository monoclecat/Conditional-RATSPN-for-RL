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
import matplotlib

from mnist_experiments.mnist_gen_train import *
import graphviz
from graphviz import nohtml
from layers import *

def plot_spn_graph(model, save_dir):
    crossprod_nodes_as_ports = True  # Otherwise the nodes don't keep their order
    t_start = time.time()
    g = graphviz.Digraph(
        'g', filename='btree.gv', comment=f"SPN",
        node_attr={'height': '.1'},
        graph_attr={'splines': 'false', 'ranksep': '2.0'},
    )
    colormap = matplotlib.colormaps['viridis']
    R = model.config.R
    cond = 0
    for layer_index in range(model.max_layer_index + 1):
        layer = model.layer_index_to_obj(layer_index)
        if layer_index == 0:
            # Leaf layer
            if False:
                means = model.means[cond].detach().cpu().numpy()  # [F, I, R]
                log_stds = model.log_stds[cond].detach().cpu().numpy()
                for r in range(means.shape[2]):
                    for f in range(means.shape[0]):
                        content = '|'.join([f'<f{layer_index}> ' for i in range(means.shape[1])])
                        g.node(f'l{layer_index}_dist_f{f}r{r}', nohtml(content))

            prod_features = model._leaf.prod._out_features
            for r in range(R):
                for f in range(prod_features):
                    with g.subgraph(name=f'clusterl{layer_index}f{f}r{r}') as c:
                        c.attr(style='filled', color='lightgrey', rank='same', ordering='in')
                        c.node_attr.update(shape='circle', style='filled', color='white')
                        [c.node(f'l{layer_index}_f{f}r{r}o{i}', ' ') for i in range(model.config.I)]
        elif isinstance(layer, CrossProduct):
            for r in range(R):
                for f in range(layer.in_features // 2):
                    content = '|'.join([f'<o{i}>X' for i in range(layer.in_channels ** 2)])
                    g.node(f'l{layer_index}_f{f}r{r}', nohtml(content), shape='record')
                    for o in range(layer.in_channels ** 2):
                        g.edge(f'l{layer_index}_f{f}r{r}:o{o}:s', f'l{layer_index - 1}_f{f * 2}r{r}o{o // layer.in_channels}:n')
                        g.edge(f'l{layer_index}_f{f}r{r}:o{o}:s', f'l{layer_index - 1}_f{f * 2 + 1}r{r}o{o % layer.in_channels}:n')
        elif isinstance(layer, Sum) and layer_index < model.max_layer_index:
            weights = layer.weights.exp().detach().cpu().numpy()
            for r in range(R):
                for f in range(layer.in_features):
                    content = '|'.join([f'<o{i}>+' for i in range(layer.out_channels)])
                    g.node(f'l{layer_index}_f{f}r{r}', nohtml(content), shape='record')
                    for o in range(layer.out_channels):
                        with g.subgraph(name=f'clusterl{layer_index}f{f}r{r}o{o}') as co:
                            for i in range(layer.in_channels):
                                weight = weights[cond, f, i, o, r]
                                color = [int(c * 256) for c in colormap.colors[int(weight * 256)]]
                                color = ''.join([f'{c:02X}' for c in color])
                                color = f'#{color}'
                                co.node(
                                    f'l{layer_index}_f{f}r{r}o{o}w{i}', f'{weight:.1f}', group=f'l{layer_index}_f{f}r{r}o{o}',
                                    shape='square', fillcolor=color, fontcolor='white',
                                )
                                g.edge(f'l{layer_index}_f{f}r{r}:o{o}:s', f'l{layer_index}_f{f}r{r}o{o}w{i}:n',
                                       fillcolor=color, color=color)
                                g.edge(f'l{layer_index}_f{f}r{r}o{o}w{i}:s', f'l{layer_index - 1}_f{f}r{r}:o{i}:n',
                                       fillcolor=color, color=color)
        elif isinstance(layer, Sum) and layer_index == model.max_layer_index:
            with g.subgraph(name=f'clusterl{layer_index}f0r0') as c:
                c.attr(style='filled', color='lightgrey')
                c.node_attr.update(shape='circle', style='filled', color='white')
                for o in range(layer.out_channels):
                    with c.subgraph(name=f'clusterl{layer_index}f0r0o{o}') as co:
                        co.node(f'l{layer_index}_f0r0o{o}', '+', group=f'l{layer_index}_f0r0o{o}')
                        for i in range(layer.in_channels):
                            co.node(
                                f'l{layer_index}_f0r0o{o}w{i}', 'W', group=f'l{layer_index}_f0r0o{o}',
                                shape='square', fontcolor='lightgray',
                            )
            n_per_rep = layer.in_channels // R
            for o in range(layer.out_channels):
                for r in range(R):
                    for i in range(layer.in_channels // R):
                        g.edge(f'l{layer_index}_f0r0o{o}:s', f'l{layer_index}_f0r0o{o}w{i + n_per_rep * r}:n')
                        g.edge(f'l{layer_index}_f0r0o{o}w{i + n_per_rep * r}:s', f'l{layer_index - 1}_f0r{r}o{i}:n')
        else:
            assert False

def bad_plot_spn_graph(model, save_dir):
    # WARNING: The nodes don't keep their order
    t_start = time.time()
    g = graphviz.Digraph(
        'g', filename='btree.gv', comment=f"SPN",
        node_attr={'height': '.1'},
        graph_attr={'splines': 'false', 'ranksep': '2.0', 'newrank': 'true'},
    )
    colormap = matplotlib.colormaps['viridis']
    R = model.config.R
    cond = 0
    for layer_index in range(model.max_layer_index + 1):
        layer = model.layer_index_to_obj(layer_index)
        if layer_index == 0:
            # Leaf layer
            if False:
                means = model.means[cond].detach().cpu().numpy()  # [F, I, R]
                log_stds = model.log_stds[cond].detach().cpu().numpy()
                for r in range(means.shape[2]):
                    for f in range(means.shape[0]):
                        content = '|'.join([f'<f{layer_index}> ' for i in range(means.shape[1])])
                        g.node(f'l{layer_index}_dist_f{f}r{r}', nohtml(content))

            prod_features = model._leaf.prod._out_features
            for r in range(R):
                for f in range(prod_features):
                    with g.subgraph(name=f'clusterl{layer_index}f{f}r{r}') as c:
                        c.attr(style='filled', color='lightgrey', rank='same', ordering='in')
                        c.node_attr.update(shape='circle', style='filled', color='white')
                        [c.node(f'l{layer_index}_f{f}r{r}o{i}', ' ') for i in range(model.config.I)]
                    for i in range(model.config.I-1):
                        g.edge(f'l{layer_index}_f{f}r{r}o{i}:e', f'l{layer_index}_f{f}r{r}o{i+1}:w')
        elif isinstance(layer, CrossProduct):
            for r in range(R):
                for f in range(layer.in_features // 2):
                    # Nodes don't keep their order!
                    with g.subgraph(name=f'clusterl{layer_index}f{f}r{r}') as c:
                        c.attr(style='filled', color='lightgrey', rank='same')
                        c.node_attr.update(shape='circle', style='filled', color='white')
                        for i in range(layer.in_channels ** 2):
                            c.node(f'l{layer_index}_f{f}r{r}o{i}', 'X')
                    for i in range(layer.in_channels ** 2 - 1):
                        g.edge(f'l{layer_index}_f{f}r{r}o{i}:e', f'l{layer_index}_f{f}r{r}o{i+1}:w')
                    for o in range(layer.in_channels ** 2):
                        g.edge(f'l{layer_index}_f{f}r{r}o{o}:s', f'l{layer_index - 1}_f{f * 2}r{r}o{o // layer.in_channels}:n')
                        g.edge(f'l{layer_index}_f{f}r{r}o{o}:s', f'l{layer_index - 1}_f{f * 2 + 1}r{r}o{o % layer.in_channels}:n')
        elif isinstance(layer, Sum) and layer_index < model.max_layer_index:
            weights = layer.weights.exp().detach().cpu().numpy()
            for r in range(R):
                for f in range(layer.in_features):
                    with g.subgraph(name=f'clusterl{layer_index}f{f}r{r}') as c:
                        c.attr(style='filled', color='lightgrey')
                        c.node_attr.update(shape='circle', style='filled', color='white')
                        for o in range(layer.out_channels):
                            with c.subgraph(name=f'clusterl{layer_index}f{f}r{r}o{o}') as co:
                                co.node(f'l{layer_index}_f{f}r{r}o{o}', '+', group=f'l{layer_index}_f{f}r{r}o{o}')
                                for i in range(layer.in_channels):
                                    weight = weights[cond, f, i, o, r]
                                    color = [int(c * 256) for c in colormap.colors[int(weight * 256)]]
                                    color = ''.join([f'{c:02X}' for c in color])
                                    color = f'#{color}'
                                    co.node(
                                        f'l{layer_index}_f{f}r{r}o{o}w{i}', f'{weight:.1f}', group=f'l{layer_index}_f{f}r{r}o{o}',
                                        shape='square', fillcolor=color, fontcolor='white',
                                    )
                                for i in range(layer.in_channels-1):
                                    co.edge(f'l{layer_index}_f{f}r{r}o{o}w{i}', f'l{layer_index}_f{f}r{r}o{o}w{i+1}')
                            for i in range(layer.in_channels):
                                c.edge(f'l{layer_index}_f{f}r{r}o{o}:s', f'l{layer_index}_f{f}r{r}o{o}w{i}:n',
                                       fillcolor=color, color=color)
                                c.edge(f'l{layer_index}_f{f}r{r}o{o}w{i}:s', f'l{layer_index - 1}_f{f}r{r}o{i}:n',
                                       fillcolor=color, color=color)
        elif isinstance(layer, Sum) and layer_index == model.max_layer_index:
            with g.subgraph(name=f'clusterl{layer_index}f0r0') as c:
                c.attr(style='filled', color='lightgrey')
                c.node_attr.update(shape='circle', style='filled', color='white')
                for o in range(layer.out_channels):
                    with c.subgraph(name=f'clusterl{layer_index}f0r0o{o}') as co:
                        co.node(f'l{layer_index}_f0r0o{o}', '+', group=f'l{layer_index}_f0r0o{o}')
                        for i in range(layer.in_channels):
                            co.node(
                                f'l{layer_index}_f0r0o{o}w{i}', 'W', group=f'l{layer_index}_f0r0o{o}',
                                shape='square', fontcolor='lightgray',
                            )
            n_per_rep = layer.in_channels // R
            for o in range(layer.out_channels):
                for r in range(R):
                    for i in range(layer.in_channels // R):
                        g.edge(f'l{layer_index}_f0r0o{o}:s', f'l{layer_index}_f0r0o{o}w{i + n_per_rep * r}:n')
                        g.edge(f'l{layer_index}_f0r0o{o}w{i + n_per_rep * r}:s', f'l{layer_index - 1}_f0r{r}o{i}:n')
        else:
            assert False

    t_delta = time.time() - t_start
    print(1)
    g.render(os.path.join(results_dir, 'test.gv'), view=True, format='png')
    cond = 0
    w_set = None
    first_sum_layer_features = None
    for i in model.sum_layer_indices:
        layer = model.layer_index_to_obj(i)
        weights = layer.weights[cond].exp().detach().cpu().numpy()
        w = np.einsum('dior -> rdoi', weights)
        if w_set is None:
            first_sum_layer_features = w.shape[1]
            w = w.flatten()
            w_set = w
        else:
            w = np.repeat(w, first_sum_layer_features // w.shape[1], axis=-1)
            w = w.flatten()
            print(1)
        print(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True,
                        help='The base directory where the directory containing the results will be saved to.')
    parser.add_argument('--model', type=str, help='Path to the model.', required=True)
    parser.add_argument('--exp_name', '-name', type=str, default='test_plot',
                        help='Experiment name. The results dir will contain it.')
    args = parser.parse_args()

    assert os.path.exists(args.model), f"The path to the model doesn't exist! {args.model1}"

    results_dir = os.path.join(args.save_dir, non_existing_folder_name(args.save_dir, f"results_{args.exp_name}"))

    model = th.load(args.model, map_location='cpu')
    if isinstance(model, CSPN):
        conditional = F.one_hot(th.as_tensor(0), num_classes=10).unsqueeze(0).float()
        model.set_params(conditional)
    plot_spn_graph(model, args.save_dir)
