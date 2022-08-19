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
import pandas as pd

if __name__ == "__main__":
    matplotlib.use('Agg')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, help='Experiment directory', required=True)
    args = parser.parse_args()
    assert os.path.exists(args.dir)

    config = None
    log = None
    cwd = os.path.realpath(args.dir)
    dir_name = os.path.split(cwd)[1]
    for filename in os.listdir(cwd):
        filename = os.fsdecode(filename)
        if filename.endswith('.csv'):
            if filename.startswith('args'):
                with open(os.path.join(cwd, filename)) as f:
                    reader = csv.DictReader(f)
                    config = [r for r in reader][0]
            elif filename.startswith('log'):
                log = pd.read_csv(os.path.join(cwd, filename))

    epochs = log['epoch'].to_numpy()[1:]
    loss = log['loss'].to_numpy()[1:]
    fig, (ax1) = plt.subplots(1, figsize=(10, 10), dpi=200)
    fig.suptitle(f"MNIST on {'RatSPN' if config['ratspn'] else 'CSPN'}")
    ax1.set_title(f"Name: {config['exp_name']} - Batch size: {config['batch_size']}")
    ax1.plot(epochs, loss)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    fig.savefig(os.path.join(args.dir, f"loss_{config['exp_name']}.png"))
    plt.close(fig)
    print(1)

