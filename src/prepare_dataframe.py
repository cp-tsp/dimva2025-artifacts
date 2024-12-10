import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import OrdinalEncoder


def delete_unwanted_data(dat_frame: pd.DataFrame):
    """
    Deleting of aborted connections and reindexing the dataframe
    :param dat_frame:
    :return: pd.DataFrame
    """
    for status in ["REJ", "RSTRH", "RSTR", "OTH", "S0", "SHR"]:
        dat_frame = dat_frame[dat_frame['conn_state'] != status]

    dat_frame = dat_frame[dat_frame['history'].str.contains("D")]
    dat_frame = pd.concat([dat_frame], ignore_index=True)

    return dat_frame.loc[:, dat_frame.notna().all(axis=0)]


def process_data(input_path: pathlib.PurePath, output_path: pathlib.PurePath, feature_group: str):
    """
    Read the input csv to modify. Returns the features desired after deleting unwanted data.
    :param input_path:
    :param output_path:
    :param feature_group:
    """
    if feature_group == "":
        list_feature_group = []
    else:
        try:
            list_feature_group = json.load(open("features.json", "r"))[f"{feature_group}"]
        except Exception as e:
            print(e)
            list_feature_group = []
            print(f"Can't find {feature_group}.\nUsing {', '.join(list_feature_group)} instead.\n")
    df = pd.read_csv(f"{input_path}")
    if not list_feature_group:
        list_feature_group = list(df.columns)
    list_feature_group.append("label")
    try:
        df = delete_unwanted_data(df)
    except Exception as e:
        print("ERROR :\n\n", e)
        raise Exception(e)
    selected_features = []

    for feature in list_feature_group:
        if feature == "tcp_flags":
            try:
                for letter in ["A", "F", "P", "R", "S",
                               "U"]:  # we remove  "C" and "E" as there bias from the use of Windows
                    df[f"tcp_flag_{letter}"] = df[[feature]].map(lambda x: 1 if letter in x else 0)
                    selected_features.append(f"tcp_flag_{letter}")
            except Exception as e:
                print(e)
                pass
            continue
        if "service" in feature:  # We map the service given by Zeek using the destination port
            services = {53: 'dns', 80: 'http', 443: 'ssl', 8572: '-', 8080: 'ssl', 8888: "ssl", 8081: "http"}
            df["service"] = list(map(lambda x: services[x], df['dst_port']))

        if "index" in feature and feature not in df.columns:  # We use an OrdinalEncoder to build the "index" features
            old_feature_name = "_".join(feature.split('_')[:-1])
            array = np.array(df[f"{old_feature_name}"]).reshape(-1, 1)
            categories_list = [
                df[
                    f"{old_feature_name}"].value_counts().index.to_list()]  # Categories sorted in the frequency order
            enc = OrdinalEncoder(categories=categories_list)
            enc.fit(array)
            df[f"{feature}"] = enc.transform(array)
        selected_features.append(feature)
    df = df.loc[:, selected_features]

    df.to_csv(f"{output_path}", index=False)


def main():
    """
    Arg parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=str, default="", required=True)
    parser.add_argument("-o", "--output-path", type=str, default="", required=True)
    parser.add_argument("-fg", "--feature-group", type=str, default="")
    args = parser.parse_args()

    input_path = pathlib.PurePath(args.input_path)
    output_path = pathlib.PurePath(args.output_path)
    feature_group = args.feature_group
    process_data(input_path, output_path, feature_group)


def run_main():
    sys.exit(main())


if __name__ == "__main__":
    run_main()
