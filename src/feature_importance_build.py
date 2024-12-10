#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import toml
from sklearn.compose import make_column_transformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from utils import estimator_builder, data_utils


def build_importance(
        prefix: str,
        suffix: str,
        n_jobs: int,
        train_csv_file_path: str,
        output_directory: str,
        method: str,
        score: str,
        multiclass: bool,
        grid_search: bool,
        ml_config_d: dict,
        random_state: int
):
    print("build_importance: start")

    print("process: reading data from CSV file")
    all_df = pd.read_csv(train_csv_file_path, index_col=False, keep_default_na=False)

    feature_df = all_df.iloc[:, all_df.columns != 'label']
    target = all_df["label"]

    malicious_labels = list(target.unique())  # Only used for multiclass experiments
    malicious_labels.remove("Benign")

    print("process: preliminary reading data and building features")
    X_train, X_val, y_train, y_val = train_test_split(feature_df, target, random_state=random_state)

    print(
        "process: reading data and building features for use case (tunnel classification or not)"
    )
    label_l = np.unique(target)
    print(f"process: tunnel_label_l: {label_l}")

    scoring_full_d = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, pos_label="CobaltStrike", zero_division="warn"),
        "recall": make_scorer(recall_score, pos_label="CobaltStrike", zero_division="warn"),
        "f1": make_scorer(f1_score, pos_label="CobaltStrike", zero_division="warn"),
        "f1-multi": make_scorer(f1_score, labels=malicious_labels, average="micro", zero_division="warn"),

    }
    scoring_permutation = scoring_full_d[score]

    print("build_importance: building estimator (RF)")

    estimator_base = estimator_builder.get_estimator(
        method,
        random_state=random_state,
    )

    if multiclass:
        estimator_base = OneVsRestClassifier(estimator_base, n_jobs=n_jobs)

    print("build_importance: building classifier")
    if grid_search:
        parameter_d = estimator_builder.get_parameter_d(
            ml_config_d, method)
        pipeline = estimator_builder.build_grid_search_cv_classifier_pipeline(
            estimator_base,
            parameter_d,
            n_jobs,
            scoring_permutation,
            int(ml_config_d['nb_folds'] / 4 + 0.5),
            refit="f1_score",
            random_state=random_state,
            verbose=1,
        )
    # n_splits = ml_config_d["nb_folds"]
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
    pipeline.fit(X_train, y_train)

    print("build_importance: MDI")

    if grid_search:
        stat_d_l = [
            data_utils.get_metric_stat_dict([
                _tree.feature_importances_[feature_i]
                for _tree in pipeline.best_estimator_.named_steps['clf']
            ]) for feature_i in range(len(list(feature_df.columns)))
        ]
    else:
        stat_d_l = [
            data_utils.get_metric_stat_dict([
                _tree.feature_importances_[feature_i]
                for _tree in pipeline.named_steps['clf']
            ]) for feature_i in range(len(list(feature_df.columns)))
        ]

    mdi_importance_mean_l = [d["mean"] for d in stat_d_l]
    mdi_importance_std_l = [d["std"] for d in stat_d_l]
    mdi_importance_sem_l = [d["sem"] for d in stat_d_l]
    mdi_importance_ci99_l = [d["ci99"] for d in stat_d_l]

    print("build_importance: permutation")

    r = permutation_importance(
        pipeline,
        X_val,
        y_val,
        n_repeats=30,
        random_state=42,
        n_jobs=n_jobs,
        scoring=scoring_permutation,
    )

    print("build_importance: r:\n", r)

    permutation_importance_mean = list(r.importances_mean)
    permutation_importance_std = list(r.importances_std)

    print('build_importance: r:\n', r)
    print(
        f"build_importance: mdi_importance_mean_l:\n{mdi_importance_mean_l}"
    )
    print(
        f"build_importance: permutation_importance_mean_l:\n{permutation_importance_mean}"
    )
    importance_d = {
        "feature_name": list(feature_df.columns),
        "mdi_importance": {
            "mean_l": mdi_importance_mean_l,
            "std_l": mdi_importance_std_l,
            "sem_l": mdi_importance_sem_l,
            "ci99_l": mdi_importance_ci99_l,
        },
        "permutation_importance": {
            "mean_l": permutation_importance_mean,
            "std_l": permutation_importance_std,
        },
    }

    basename = Path(train_csv_file_path).stem
    if prefix:
        prefix = prefix + "_"
    if suffix:
        suffix = "_" + suffix
    importance_file_path = Path(output_directory, f"{prefix}{basename}{suffix}_{method}_{score}_fi.json")
    print(importance_file_path)
    with open(importance_file_path, "w", encoding="utf-8") as fp:
        json.dump(importance_d, fp, indent=2)

    print("build_importance: end")


def main():
    """
    Arg parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="CSV files for train",
                        default=[])
    parser.add_argument("-j", "--n-jobs", type=int, default=1)
    parser.add_argument("-o", "--output-directory", type=str, default="", required=True)
    parser.add_argument("-l", "--method", type=str, default="rf")
    parser.add_argument("--multiclass", action="store_true")  # Used for testing multiclass strategies
    parser.add_argument("-s", "--suffix", type=str, default="")
    parser.add_argument("-p", "--prefix", type=str, default="")
    parser.add_argument("--score", type=str, default="f1_score")
    parser.add_argument("--seed", type=int, default=None, help="Seed used in ML")
    parser.add_argument("--gs", action="store_true")
    parser.add_argument("-c", "--ml-config-file-path", type=str, default="")

    args = parser.parse_args()
    method = args.method
    n_jobs = args.n_jobs
    output_directory = args.output_directory
    score = args.score
    prefix = args.prefix
    suffix = args.suffix
    random_state = args.seed
    train_csv_file_path = args.input
    multiclass = args.multiclass
    grid_search = args.gs
    ml_config_file_path = args.ml_config_file_path

    if multiclass:
        raise Exception("Not supported yet")

    print(f"train_csv_file_path_l: {train_csv_file_path}")
    print(f"n_jobs: {n_jobs}")
    print(f"output_directory: {output_directory}")
    print(f"suffix: {suffix}")

    with open(ml_config_file_path, "r", encoding="utf-8") as content_file:
        toml_string = content_file.read()
    ml_config_d = toml.loads(toml_string)
    print(f"ml_config_d: {ml_config_d}")

    build_importance(
        prefix,
        suffix,
        n_jobs,
        train_csv_file_path,
        output_directory,
        method,
        score,
        multiclass,
        grid_search,
        ml_config_d,
        random_state
    )


def run_main():  # pylint: disable=invalid-name
    sys.exit(main())


if __name__ == "__main__":
    run_main()
