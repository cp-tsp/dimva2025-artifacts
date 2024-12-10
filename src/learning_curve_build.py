#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import toml
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from utils import data_utils
from utils import estimator_builder


def process(
        method_l: list[str],
        ml_config_d: dict,
        score: str,
        prefix: str,
        suffix: str,
        n_jobs: int,
        csv_file_path: str,
        output_directory: Path,
        multiclass,
        grid_search,
        random_state: int = 0,
):
    print("process: start")

    n_splits = ml_config_d["nb_folds"]

    print("process: reading data from CSV file")
    all_df = pd.read_csv(csv_file_path, index_col=False, keep_default_na=False).dropna(axis=1)

    feature_df = all_df.iloc[:, all_df.columns != 'label']
    target = all_df["label"]

    malicious_labels = list(target.unique())  # Only used for multiclass experiments
    malicious_labels.remove("Benign")

    print("process: feature_df column nb: ", len(feature_df.columns))
    print("process: feature_df line nb: ", len(feature_df))
    if len(feature_df.columns) < 10:
        print("process: feature_df columns: ", feature_df.columns)

    del all_df

    scoring_full_d = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, pos_label="CobaltStrike"),
        "recall": make_scorer(recall_score, pos_label="CobaltStrike"),
        "f1": make_scorer(f1_score, pos_label="CobaltStrike"),
        "f1_multi": make_scorer(f1_score, labels=malicious_labels, average="micro"),
    }
    scoring_learning_curve = scoring_full_d[score]
    data_d: dict = {"time": {}}

    dt = datetime.datetime.now()
    data_d["time"]["start_time"] = dt.strftime("%Y-%m-%d_%H:%M:%S")

    for method in method_l:
        print(f"\n\nmethod: {method}")

        method_performance_d = {}

        print("process: building classifier")
        estimator_base = estimator_builder.get_estimator(
            method, random_state=random_state)

        if multiclass:
            estimator_base = OneVsRestClassifier(estimator_base, n_jobs=n_jobs)

        if grid_search:
            parameter_d = estimator_builder.get_parameter_d(
                ml_config_d, method)
            pipeline = estimator_builder.build_grid_search_cv_classifier_pipeline(
                estimator_base,
                parameter_d,
                n_jobs,
                scoring_learning_curve,
                int(ml_config_d['nb_folds'] / 4 + 0.5),
                refit="f1_score",
                random_state=random_state,
                verbose=1,
            )
        elif method == "ramos":
            ordinal_encoder = make_column_transformer(
                (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                categories=[feature_df["ip_protocol"].value_counts().index.to_list()]),
                 ["ip_protocol"],
                 ),
                (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                categories=[feature_df["service"].value_counts().index.to_list()]),
                 ["service"],
                 ),
                (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                categories=[feature_df["history"].value_counts().index.to_list()]),
                 ["history"],
                 ),
                remainder='passthrough',
            )
            pipeline = Pipeline(steps=[('enc', ordinal_encoder),
                                       ('clf', estimator_base)])  # No standard scaler for Ramos et al. methodology
        else:
            pipeline = estimator_builder.build_base_estimator(estimator_base)

        cv = StratifiedKFold(n_splits=n_splits,
                             shuffle=True,
                             random_state=random_state)
        train_size_a_log = np.logspace(np.log10(50 / len(target)), 0, 20)
        print(f"process: train_sizes (length {len(train_size_a_log)}): {train_size_a_log}")

        print("process: learning_curve")
        train_size_s_l, train_scores, test_scores, fit_times, score_times = learning_curve(
            estimator=pipeline,
            X=feature_df,
            y=target,
            train_sizes=train_size_a_log,
            cv=cv,
            scoring=scoring_learning_curve,
            shuffle=True,
            random_state=random_state,
            error_score="raise",
            return_times=True,
            n_jobs=n_jobs,
            verbose=5
        )
        print("process: done")
        print(fit_times)
        print(score_times)
        train_size_l = [int(s) for s in list(train_size_s_l)]
        train_f1_score_l_l = train_scores.tolist()
        test_f1_score_l_l = test_scores.tolist()

        print(f"process: train_size_l: {train_size_l}")
        print(f"process: train_f1_score_l_l: {train_f1_score_l_l}")
        print(f"process: test_f1_score_l_l: {test_f1_score_l_l}")

        method_performance_d["res"] = {
            train_size: {
                "train_f1_score":
                    data_utils.get_metric_stat_dict(train_f1_score_l),
                "test_f1_score":
                    data_utils.get_metric_stat_dict(test_f1_score_l),
            }
            for train_size, train_f1_score_l, test_f1_score_l in zip(
                train_size_l, train_f1_score_l_l, test_f1_score_l_l)
        }

        method_performance_d["train_size_l"] = train_size_l
        method_performance_d["train_scores"] = train_f1_score_l_l
        method_performance_d["test_scores"] = test_f1_score_l_l

        data_d[method] = method_performance_d
        basename = Path(csv_file_path).stem
        if prefix:
            prefix = prefix + "_"
        if suffix:
            suffix = "_" + suffix
        learning_curve_file_path = Path(output_directory, f"{prefix}{basename}{suffix}_{method}_{score}_lc.json")
        print(f"learning_curve_file_path: {learning_curve_file_path}")
        tunnel_performance_s = json.dumps(data_d, indent=2)
        with open(learning_curve_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(tunnel_performance_s)

    print(f"process: finished at {datetime.datetime.now()}")

    print("process: end")


def main():
    """
    Arg parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ml-config-file-path", type=str, default="")
    parser.add_argument("-i", "--csv-file-path", type=str, default="", required=True)
    parser.add_argument("-j", "--n-jobs", type=int, default=1)
    parser.add_argument("-l", "--method-l", nargs="+", type=str, help="<Required> Set flag", required=True)
    parser.add_argument("-o", "--output-directory", type=str, default="", required=True)
    parser.add_argument("--multiclass", action="store_true")  # Used for testing multiclass strategies
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--score", type=str, default="f1")
    parser.add_argument("--seed", type=int, default=None, help="Seed used in ML")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--gs", action="store_true")

    args = parser.parse_args()

    csv_file_path = args.csv_file_path
    n_jobs = args.n_jobs
    ml_config_file_path = args.ml_config_file_path
    output_directory = args.output_directory
    score = args.score
    prefix = args.prefix
    suffix = args.suffix
    method_l = args.method_l
    seed = args.seed
    multiclass = args.multiclass
    grid_search = args.gs

    print(f"csv_file_path: {csv_file_path}")
    print(f"n_jobs: {n_jobs}")
    print(f"ml config file path: {ml_config_file_path}")
    print(f"output_directory: {output_directory}")
    print(f"score: {score}")
    print(f"prefix: {prefix}")
    print(f"suffix: {suffix}")
    print(f"seed: {seed}")

    with open(ml_config_file_path, "r", encoding="utf-8") as content_file:
        toml_string = content_file.read()
    ml_config_d = toml.loads(toml_string)
    print(f"ml_config_d: {ml_config_d}")

    process(
        method_l,
        ml_config_d,
        score,
        prefix,
        suffix,
        n_jobs,
        csv_file_path,
        output_directory,
        multiclass,
        grid_search,
        random_state=seed
    )


def run_main():  # pylint: disable=invalid-name
    sys.exit(main())


if __name__ == "__main__":
    run_main()
