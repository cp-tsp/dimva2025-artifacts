"""
Plot the labels figures
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

with open("config.json", "r") as conf:
    config = json.load(conf)
COLORS = config["colors"]
LABELS = config["labels"]
MARKERS = config["markers"]
HATCHS = config["hatchs"]
NUMBER = len(LABELS)


def main():
    """
    Arg parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=str, default="", required=True)
    parser.add_argument("-p", "--protocol", type=str, default="", required=True)
    parser.add_argument("-fi", "--feature-importance", action="store_true",
                        help="If activated the feature importance labels will be generated")
    args = parser.parse_args()

    output_path = pathlib.PurePath(args.output_path)
    feature_importance = args.feature_importance
    protocol = args.protocol
    generate_labels(feature_importance, output_path, protocol)


def generate_labels(feature_importance, output_path, protocol):
    x = np.linspace(0, 100, NUMBER)
    y = [[i * j for j in range(NUMBER)] for i in range(NUMBER)]
    fig, ax = plt.subplots()
    if protocol == "http":
        nu, limit = 0, 4
        figsize = (36, 6)
        NCOLS = 2
    elif protocol == "https":
        nu, limit = 4, 9
        figsize = (37, 7)
        NCOLS = 2
    elif protocol == "dns":
        nu, limit = 9, 11
        figsize = (16, 4)
        NCOLS = 1
    else:
        raise "Error wrong protocol"

    while nu < limit:
        if feature_importance:
            ax.bar(x, y[nu], color=COLORS[nu % len(COLORS)], linewidth=10, label=LABELS[nu],
                   alpha=.99)
            ax.plot(x, y[nu], color=COLORS[nu], linewidth=0, label=LABELS[nu],
                    marker=MARKERS[nu % len(MARKERS)], markersize=16)

        else:
            ax.plot(x, y[nu], color=COLORS[nu % len(COLORS)], linewidth=15, label=LABELS[nu],
                    marker=MARKERS[nu % len(MARKERS)], markersize=25)  # ,mew=mew)
        nu += 1

    ax.legend(loc="best", ncols=NUMBER, fontsize=40)
    label_params = ax.get_legend_handles_labels()
    handles = [(label_params[0][i], label_params[0][len(label_params[0]) // 2 + i]) for i in
               range(len(label_params[0]) // 2)]

    if feature_importance:
        figl, axl = plt.subplots(figsize=figsize)
        axl.axis(False)
        axl.legend(handles, label_params[1], loc="center", markerscale=6, prop={"size": 110},
                   ncols=NCOLS, handler_map={tuple: HandlerTuple(ndivide=1.6)}, framealpha=0, columnspacing=0.4)
        # works with a float and returns a better result with ndivide=1.6
    else:
        figl, axl = plt.subplots(figsize=figsize)
        axl.axis(False)
        axl.legend(*label_params, loc="center", markerscale=4, prop={"size": 110}, ncols=NCOLS, framealpha=0,
                   columnspacing=0.4)

    if feature_importance:
        figl.savefig(pathlib.PurePath(output_path, f"labels_{protocol}_fi.pdf"))
    else:
        figl.savefig(pathlib.PurePath(output_path, f"labels_{protocol}.pdf"))


def run_main():
    sys.exit(main())


if __name__ == "__main__":
    run_main()
