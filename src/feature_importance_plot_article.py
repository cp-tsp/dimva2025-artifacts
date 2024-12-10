import argparse
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import ScaledTranslation

from utils import plot_utils

with open("config.json", "r") as conf:
    config = json.load(conf)
HATCHS = config["hatchs"]
COLORS = config["colors"]
MARKERS = config["markers"]


def plot_mdi(
        mdi_importance_mean_l: list[float],
        mdi_importance_eb_l: list[float],
        feature_name,
        final_order: list,
        i,
        lab,
        color,
        marker,
        hatch,
        n_files,
        width,
):
    """
    Plot the MDI for the given features of a file.
    :param mdi_importance_mean_l:
    :param mdi_importance_eb_l:
    :param feature_name:
    :param final_order:
    :param i:
    :param lab:
    :param color:
    :param marker:
    :param hatch:
    :param n_files:
    :param width:
    :return:
    """
    indexes = [feature_name.index(i) for i in final_order]

    mdi_importance_mean_l_first = [
        mdi_importance_mean_l[i] for i in indexes
    ]
    mdi_importance_eb_l_first = [
        mdi_importance_eb_l[i] for i in indexes
    ]

    x = range(len(final_order))
    x = [i + k * width * (n_files + 1) for k in x]
    print("mdi: x: " + str(x))
    print(f"mdi: mdi_importance_mean_l_first: {mdi_importance_mean_l_first}")
    marker = marker[:len(x)]
    bar = plt.bar(x, mdi_importance_mean_l_first, yerr=mdi_importance_eb_l_first, width=width, label=lab,
                  color=color, alpha=.99, hatch=hatch)
    plt.bar_label(bar, labels=marker, fontsize=30, color=color, label_type="edge", padding=8)
    plt.grid(axis="y")


def plot_files(input_json_file_path: list, nb_feature: int, hatching: bool, arg_ylim: float):
    """
    Prepare the figure for all the files given in input
    :param arg_ylim:
    :param input_json_file_path:
    :param nb_feature:
    :param hatching:
    """
    if "http" in input_json_file_path[0].name:
        if "https" in input_json_file_path[0].name:
            color_nb = 4
            if "generic" not in input_json_file_path[0].name:
                color_nb += 3
        else:
            color_nb = 0
            if "generic" not in input_json_file_path[0].name:
                color_nb += 2

    else:
        color_nb = 9
    i = 0
    feature_group = input_json_file_path[0].stem.split('_')[-4]

    try:
        params = config[f"netflow{feature_group}"]
        fig_w, fig_h = params["width"], params["height"]
        font_size, font_size_ticks = params["font_size"], params["font_size_ticks"]
        if arg_ylim:
            ylim = arg_ylim
        else:
            ylim = params["ylim"]
        if feature_group == "v9e":
            if "dns" in input_json_file_path[0].name:
                fig_w, fig_h = 18, 8
    except Exception as e:
        print(f"{e} : Configuration not found for this feature group. Using default parameters.")
        fig_w, fig_h = 16, 8
        font_size, font_size_ticks = 60, 50
        ylim = 1

    if arg_ylim:
        ylim = arg_ylim
    feature_mean_df = pd.DataFrame()
    for input_json in input_json_file_path:
        if os.path.exists(input_json):
            if os.path.getsize(input_json) == 0:
                print(f"Failure: {input_json} is empty")
                sys.exit()
        else:
            print(f"Failure: {input_json} does not exists")
            sys.exit()

        with open(input_json, "r") as file:
            data_d = json.load(file)

        if not list(feature_mean_df.columns):
            for f in data_d["feature_name"]:
                feature_mean_df.loc[:, f] = []

        feature_mean_df = pd.concat([pd.DataFrame([data_d["mdi_importance"]["mean_l"]], columns=data_d["feature_name"]),
                                     feature_mean_df], ignore_index=True)

    # We order the different features selected
    test_d = feature_mean_df.mean().sort_values(ascending=False)
    final_order = list(feature_mean_df.mean().sort_values(ascending=False).keys())[
                  :len([test_d.get(x) for x in test_d if x > 0.04])]     # keep only feature with MDI > 0.01

    feature_name_l_label = [
        plot_utils.build_short_feature_name_for_display(feature_name)
        for feature_name in final_order]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    n_files = len(input_json_file_path)
    width = 42  # value doesn't matter
    for input_json in input_json_file_path:
        with open(input_json, "r") as file:
            data_d = json.load(file)

        feature_name = data_d["feature_name"]
        mdi_importance_mean_l = data_d["mdi_importance"]["mean_l"]
        mdi_importance_eb_l = data_d["mdi_importance"]["ci99_l"]

        label_l = input_json.stem.split("_")
        try:
            lab = f"{label_l[1]}/{label_l[3]}"
        except Exception as e:
            print(e)
            lab = f"{label_l[0]}"
        color = COLORS[color_nb % len(COLORS)]  # 4
        marker = [MARKERS[color_nb % len(MARKERS)] for i in range(len(mdi_importance_mean_l))]
        if hatching:
            hatch = HATCHS[color_nb % len(HATCHS)]
        else:
            hatch = ""

        print("mdi: start")

        print("mdi: nb_feature_used: " + str(nb_feature))

        print("mdi_importance_mean_l: " + str(mdi_importance_mean_l))
        print("mdi: font_size: " + str(font_size))

        plot_mdi(
            mdi_importance_mean_l,
            mdi_importance_eb_l,
            feature_name,
            final_order,
            i,
            lab,
            color,
            marker,
            hatch,
            n_files,
            width,
        )
        i += width
        color_nb += 1

    ticks = list(range(len(final_order)))
    ticks = [((n_files - 1) * 0.5 + k * (n_files + 1)) * width for k in ticks]
    plt.grid(True, axis="y")
    plt.ylim(0, ylim)

    plt.xticks(ticks, feature_name_l_label, rotation=65, ha="right", rotation_mode='anchor', fontsize=font_size_ticks)
    plt.yticks(fontsize=font_size_ticks)
    dx, dy = -5, 2
    offset = ScaledTranslation(dx / fig.dpi, dy / fig.dpi, fig.dpi_scale_trans)
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    if "http" in input_json_file_path[0].name:
        if "https" in input_json_file_path[0].name:
            pass
        else:
            pass  # ylabel here for MDI in HTTP figures
    else:
        pass
        # ylabel here for MDI in DNS figures
    plt.ylabel("MDI", fontsize=font_size)


def main(input_json_file_path, output_directory_path, arg_ylim, nb_feature=10, hatching=False):
    input_json_file_path = [Path(input_json_path) for input_json_path in input_json_file_path]
    output_directory_path = Path(output_directory_path)

    plot_files(
        input_json_file_path,
        nb_feature=nb_feature,
        hatching=hatching,
        arg_ylim=arg_ylim)

    figure_path = Path(output_directory_path)
    plt.savefig(figure_path, bbox_inches="tight")
    plt.clf()
    plt.close()


def run_main():  # pylint: disable=invalid-name
    """
    Arg parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-json-file-path", nargs="+", type=str, required=True)
    parser.add_argument("-o", "--output-directory-path", type=str, required=True)
    parser.add_argument("-n", "--nb-feature", type=int, default=10)
    parser.add_argument("--ylim", type=float)
    parser.add_argument("--hatching", action='store_true')
    args = parser.parse_args()
    sys.exit(main(input_json_file_path=args.input_json_file_path,
                  output_directory_path=args.output_directory_path,
                  nb_feature=args.nb_feature,
                  hatching=args.hatching,
                  arg_ylim=args.ylim))


if __name__ == "__main__":
    run_main()
