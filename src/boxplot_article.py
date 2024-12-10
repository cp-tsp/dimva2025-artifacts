import argparse
import json
import os
import sys
from pathlib import Path

from matplotlib import pyplot as plt

with open("config.json", "r") as conf:
    config = json.load(conf)
COLORS = config["colors"]
MARKERS = config["markers"]
LABELS = config["labels_short"]


def plot_boxplot(
        values_to_plot: list[float],
        basename_wo_ext,
        color,
        i,
):
    label_l = basename_wo_ext.split("_")

    if i > 2:  # To add a separation after the 3 first boxplot
        i += 2

    if "b" not in label_l[3]:
        label_l[3] = "[46]"
    else:
        label_l[3] = f"NF {label_l[3].strip('b')}"
    plt.boxplot(values_to_plot, positions=[0.7 * i], widths=0.4, boxprops={'color': color},
                labels=[f"{label_l[3]}"], showfliers=False)
    plt.xticks(rotation=55, ha="right", rotation_mode='anchor')


def plot_file(method_data_d, basename_wo_ext, color, method, i):
    print(f"plot_task: method: {method}")
    res_data_d = method_data_d[method]["res"]
    values_to_plot = res_data_d[list(res_data_d.keys())[-1]]["test_f1_score"]["values"]
    plot_boxplot(values_to_plot, basename_wo_ext, color, i)


def process(json_result_file_path, crossed, y_ticks):
    # fontsize = 28
    fontsize_ticks = 16

    # plt.ylabel("F1 score", fontsize=fontsize, labelpad=5)
    i = 0
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
            if "ramos" in basename_wo_ext:
                color = COLORS[0]
            else:
                color = COLORS[1]

        else:
            color = COLORS[i % len(COLORS)]

        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize_ticks)
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize_ticks)

        plt.grid(ls="--", axis="y")

        if not y_ticks:
            plt.tick_params(labelleft=False)  # remove the ticks

        if "ramos" in basename_wo_ext:
            method = "ramos"
        elif "rf" in basename_wo_ext:
            method = "rf"
        elif "tre" in basename_wo_ext:
            method = "tre"
        else:
            method = "rf"
        plot_file(result_data_d, basename_wo_ext, color, method,  i)

        i += 1
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)


def main(input_json_file_path, output_directory_path, crossed, y_min, y_ticks):
    input_json_file_path = [Path(input_json_path) for input_json_path in
                            input_json_file_path]
    output_directory_path = Path(output_directory_path)
    feature_group = input_json_file_path[0].stem.split('_')[-4]
    try:
        params = config[f"netflow{feature_group}"]
        if not y_min:
            y_min = params["ymin"]
        else:
            pass

    except Exception as e:
        print(f"{e} : Configuration not found for this feature group. Using default parameters.")
    plt.figure(1, figsize=(1.5, 2))  # default 5,3
    plt.ylim(y_min, 1.001)

    for input_json in input_json_file_path:
        if os.path.exists(input_json):
            if os.path.getsize(input_json) == 0:
                print(f"Failure: {input_json} is empty")  # forgot f in the original script
                sys.exit()
        else:
            print(f"Failure: {input_json} does not exists")
            sys.exit()

    process(input_json_file_path, crossed, y_ticks)  # , baxes)
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
    parser.add_argument("--crossed", action="store_true")
    parser.add_argument("--ymin", type=float)
    parser.add_argument("--y_ticks", action="store_true")

    args = parser.parse_args()
    sys.exit(main(input_json_file_path=args.input_json_file_path,
                  output_directory_path=args.output_directory_path,
                  crossed=args.crossed,
                  y_min=args.ymin,
                  y_ticks=args.y_ticks))


if __name__ == "__main__":
    run_main()
