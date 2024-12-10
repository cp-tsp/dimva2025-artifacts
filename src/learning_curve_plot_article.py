#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

with open("config.json", "r") as conf:
    config = json.load(conf)
COLORS = config["colors"]
MARKERS = config["markers"]


def plot_learning_curve_naive(
        train_size_l: list[int],
        test_f1_score_mean_l: list[float],
        test_f1_score_eb_l: list[float],
        basename_wo_ext,
        color,
        marker,
        method,
        fontsize,
        compare,
        crossed,
        alpha: float = 0.1,
):
    label_l = basename_wo_ext.split("_")
    if compare:
        if len(label_l) >= 5:
            lab = f"{label_l[0]} - {label_l[4]}"
        else:
            lab = f"{label_l[0]}"
    elif crossed:
        lab = f"Benign:{label_l[1]};Train:{label_l[3]};Test:{label_l[5]}"
        if label_l[6] in ['f1', 'precision', "recall"]:
            lab = lab + f";{label_l[6]}"
        elif label_l[7] in ['f1', 'precision', "recall"]:
            lab = lab + f";{method};{label_l[7]}"
        else:
            pass
            plt.ylabel("F1 score", fontsize=fontsize, labelpad=80)
    else:
        lab = f"{label_l[1]}/{label_l[3]}"
    if len(label_l) > 10:
        lab = lab + f";{label_l[8]}"
    plt.plot(train_size_l,
             test_f1_score_mean_l,
             label=lab,
             color=color,
             marker=marker,
             markersize=4.5)

    plt.fill_between(
        train_size_l,
        np.array(test_f1_score_mean_l) + np.array(test_f1_score_eb_l),
        np.array(test_f1_score_mean_l) - np.array(test_f1_score_eb_l),
        color=color,
        alpha=alpha,
    )
    plt.semilogx(train_size_l, train_size_l)
    plt.ylabel("F1 score", fontsize=fontsize, labelpad=10)


def plot_file(method_data_d: dict, basename_wo_ext, method, color, marker, fontsize,
              compare, crossed):
    """
    Read values to be plotted
    :param method:
    :param method_data_d:
    :param basename_wo_ext:
    :param color:
    :param marker:
    :param fontsize:
    :param compare:
    :param crossed:
    """

    print(f"plot_task: method: {method}")

    train_size_l = method_data_d[method]["train_size_l"]

    res_data_d = method_data_d[method]["res"]

    train_size_s_l = list(res_data_d.keys())

    test_f1_score_mean_l = [
        res_data_d[train_size_s]["test_f1_score"]["mean"]
        for train_size_s in train_size_s_l
    ]
    test_f1_score_macro_eb_l = [
        res_data_d[train_size_s]["test_f1_score"]["ci95"]
        for train_size_s in train_size_s_l
    ]

    plot_learning_curve_naive(
        train_size_l,
        test_f1_score_mean_l,
        test_f1_score_macro_eb_l,
        basename_wo_ext,
        color,
        marker,
        method,
        fontsize,
        compare,
        crossed
    )


def process(json_result_file_path, compare, crossed):
    """
    Prepare the figure for all the files given in input
    :param json_result_file_path:
    :param compare:
    :param crossed:
    """
    print("process: start")
    fontsize = 30
    fontsize_ticks = 28
    if "http" in json_result_file_path[0].name:
        if "https" in json_result_file_path[0].name:
            i = 4
        else:
            i = 0
            plt.ylabel("F1 score", fontsize=fontsize, labelpad=20)
    else:
        i = 9

    for file in json_result_file_path:
        with open(file, "r", encoding="utf-8") as content_file:
            json_result_string = content_file.read()
        result_data_d = json.loads(json_result_string)

        basename_wo_ext = file.stem

        print(f"process: result_data_d.keys(): {result_data_d.keys()}")
        if crossed:
            date = basename_wo_ext.split('_')[5]
            i = ["2023-01-12", "2023-01-18", "2023-01-31", "2023-05-22",
                 "2023-05-23", "2023-07-12", "2023-09-28", "2023-10-03"].index(date)
            marker = MARKERS[i % len(MARKERS)]
            if "ramos" in basename_wo_ext:
                color = COLORS[0]
            else:
                color = COLORS[1]

        else:
            color = COLORS[i % len(COLORS)]
            marker = MARKERS[i % len(MARKERS)]

        plt.xlabel("Nb training flows", fontsize=fontsize,
                   labelpad=10, loc="left")
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize_ticks)
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize_ticks)

        plt.grid(ls="--")

        if "ramos" in basename_wo_ext:
            method = "ramos"
        elif "rf" in basename_wo_ext:
            method = "rf"
        else:
            print("No compatible method found in filenames. Trying with default Random Forest method.")
            method = "rf"
        plot_file(result_data_d, basename_wo_ext, method, color, marker, fontsize,
                  compare, crossed)
        i += 1
        print("process: end")


def main(input_json_file_path, output_directory_path, y_lim=1, compare=False, crossed=False, ):
    input_json_file_path = [Path(input_json_path) for input_json_path in
                            input_json_file_path]
    output_directory_path = Path(output_directory_path)

    plt.figure(1, figsize=(7, 3))  # default 7,3
    plt.ylim(0, y_lim)
    plt.figure(1, figsize=(5, 3))  # default 5,3

    for input_json in input_json_file_path:
        if os.path.exists(input_json):
            if os.path.getsize(input_json) == 0:
                print(f"Failure: {input_json} is empty")
                sys.exit()
        else:
            print(f"Failure: {input_json} does not exists")
            sys.exit()

    process(input_json_file_path, compare, crossed)
    print(f"OUTPUT_PATH : {output_directory_path}")
    plt.savefig(output_directory_path, bbox_inches="tight")
    plt.clf()
    plt.close()


def run_main():
    """
    Arg parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-json-file-path", nargs="+", type=str, default="")
    parser.add_argument("-o", "--output-directory-path", type=str, required=True)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--crossed", action="store_true")
    parser.add_argument("--ylim", type=float, default=1)
    args = parser.parse_args()
    sys.exit(main(input_json_file_path=args.input_json_file_path,
                  output_directory_path=args.output_directory_path,
                  compare=args.compare,
                  crossed=args.crossed,
                  y_lim=args.ylim))


if __name__ == "__main__":
    run_main()
