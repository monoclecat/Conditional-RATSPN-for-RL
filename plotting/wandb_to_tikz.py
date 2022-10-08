import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, nargs='+', help='Directory containing experiment directories', required=True)
    args = parser.parse_args()

    for cwd in args.dir:
        cwd = cwd.replace('\'', '')
        cwd = os.path.realpath(cwd)
        assert os.path.exists(cwd), f"Path {cwd} doesn't exist!"
        print(f"Reading from {cwd}")
