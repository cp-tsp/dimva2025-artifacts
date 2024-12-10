import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# https://medium.com/@literallywords/sklearn-identity-transformer-fcc18bac0e98
class IdentityTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, input_array):

        return input_array * 1


class RegexFilterer(BaseEstimator, TransformerMixin):
    """Adapted from code by @zacstewart
    https://github.com/zacstewart/kaggle_seeclickfix/blob/master/estimator.py
    Also see Zac Stewart's excellent blogpost on pipelines:
    http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
    """

    def __init__(self, column_regex):
        # print("RegexCountVectorizer: transform: self.column_regex:",self.column_regex)
        self.column_regex = column_regex

    def transform(self, df: pd.DataFrame):
        df_filtered = df.filter(regex=self.column_regex)
        return df_filtered

    def fit(self, *_):
        return self


class StringBuilder(BaseEstimator, TransformerMixin):
    """Adapted from code by @zacstewart
    https://github.com/zacstewart/kaggle_seeclickfix/blob/master/estimator.py
    Also see Zac Stewart's excellent blogpost on pipelines:
    http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
    """

    def __init__(self):
        pass

    def transform(self, matrix):
        # print("StringBuilder: transform: start")
        # print("MatrixCountVectorizer: transform: matrix:",matrix)
        df = pd.DataFrame(matrix)
        # print("StringBuilder: transform: df.shape: ",df.shape)
        # print("StringBuilder: transform: casting to int")
        df = df.astype(np.uint64)
        # print("StringBuilder: transform: casting to str")
        df = df.astype(str)
        # print("StringBuilder: transform: df:",df.head(5))
        # print("StringBuilder: transform: building one string for all columns")
        # print("StringBuilder: transform: aggregation")
        str_l = df.agg("  ".join, axis=1)
        # print("StringBuilder: transform: len(str_l): ",len(str_l))
        # print("StringBuilder: transform: str_l: ",str_l)
        # print("StringBuilder: transform: end")
        return str_l

    def fit(self, *_):
        return self


def estimator_has_randomness(method: str):
    if method in ("knn", "gnb"):
        return False
    else:
        return True


method_config_key = {
    "ab": "AB_PARAMS",
    "gb": "GB_PARAMS",
    "gnb": "GNB_PARAMS",
    "knn": "KNN_PARAMS",
    "lr": "LR_PARAMS",
    "lsvc": "LSVC_PARAMS",
    "mlp": "MLP_PARAMS",
    "rf": "RFC_PARAMS",
    "sgd-lr": "SGD_LR_PARAMS",
    "sgd-sv": "SGD_SV_PARAMS",
    "svc": "SVC_PARAMS",
    "tre": "TRE_PARAMS",
    "xg": "XG_PARAMS",
}


def get_parameter_d(
        ml_config_d: dict,
        method: str,
):
    return ml_config_d[method_config_key[method]]


def parameter_d_with_single_value(d):
    return not any(len(v) > 1 for v in d.values())


def parameter_d_for_estimator(d):
    assert parameter_d_with_single_value(d)
    assert all(len(v) == 1 for v in d.values())
    return {k: l[0] for k, l in d.items()}


def get_estimator(
        method: str,
        random_state: int = None,
):
    clf = None
    if method == "ab":
        clf = AdaBoostClassifier(random_state=random_state)
    elif method == "gb":
        clf = GradientBoostingClassifier(random_state=random_state)
    elif method == "gnb":
        # No random
        clf = GaussianNB()
    elif method == "knn":
        # No random
        clf = KNeighborsClassifier(n_jobs=1)
    elif method == "lr":
        clf = LogisticRegression(n_jobs=1, random_state=random_state)
    elif method == "lsvc":
        # If solver is sag, saga or liblinear
        clf = LinearSVC(random_state=random_state)
    elif method == "mlp":
        clf = MLPClassifier(random_state=random_state)
    elif method == "rf":
        clf = RandomForestClassifier(n_jobs=1, random_state=random_state)
    elif method == "ramos":
        clf = RandomForestClassifier(n_jobs=1, random_state=random_state, max_depth=15, n_estimators=30)
    elif method == "sgd-lr":
        if random_state is None:
            shuffle = False
        else:
            shuffle = True
        clf = SGDClassifier(n_jobs=1,
                            shuffle=shuffle,
                            random_state=random_state)
    elif method == "sgd-sv":
        if random_state is None:
            shuffle = False
        else:
            shuffle = True
        clf = SGDClassifier(n_jobs=1,
                            shuffle=shuffle,
                            random_state=random_state)
    elif method == "svc":
        clf = SVC(random_state=random_state)
    elif method == "tre":
        clf = DecisionTreeClassifier(random_state=random_state)
    elif method == "xg":
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            random_state=random_state,
            use_label_encoder=False,
        )

    return clf


def build_base_estimator(estimator):
    print("estimator_builder: build_base_estimator: start")
    st = StandardScaler()
    # pca = PCA(n_components=2)

    # pipeline = Pipeline(steps=[("scaler", st), ("pca", pca), ("clf", estimator)])
    # pipeline = Pipeline([("clf", estimator)])
    pipeline = Pipeline(steps=[("scaler", st), ("clf", estimator)])

    print("estimator_builder: build_base_estimator: end")

    return pipeline


def build_grid_search_cv_classifier_pipeline(
        classifier,
        parameter_d: dict,
        n_jobs: int,
        scoring: dict,
        n_splits: int,
        refit: str,
        random_state: int = None,
        verbose: int = 0,
):
    print("estimator_builder: build_gridsearchcv_classifier_pipeline: start")

    pipeline = build_base_estimator(classifier)

    parameter_d_mod = {"clf__" + k: v for k, v in parameter_d.items()}

    cv = StratifiedKFold(n_splits=n_splits,
                         shuffle=True,
                         random_state=random_state,
                         )

    grid_search_estimator = GridSearchCV(
        pipeline,
        parameter_d_mod,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv,
        refit=refit,
        verbose=verbose,
        error_score="raise"
    )

    print("estimator_builder: build_grid_search_cv_classifier_pipeline: end")

    return grid_search_estimator
