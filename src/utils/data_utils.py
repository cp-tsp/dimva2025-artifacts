import numpy as np
import pandas as pd
import scipy


def float_to_uint_int_float(s):

    if s.astype(np.uint64) == s:
        return pd.to_numeric(s, downcast="unsigned")
    elif s.astype(np.int64) == s:
        return pd.to_numeric(s, downcast="signed")
    elif s.astype(np.float32) == s:
        return pd.to_numeric(s, downcast="float")
    elif s.astype(str) == s:
        return s
    else:
        return s


# from https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
# see also: https://en.wikipedia.org/wiki/Confidence_interval#Example
def confidence_interval(data: list, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return h


def get_metric_stat_dict(data: list):
    mean = np.mean(data)
    std = np.std(data)
    sem = scipy.stats.sem(data)
    ci90 = confidence_interval(data, confidence=0.90)
    ci95 = confidence_interval(data, confidence=0.95)
    ci99 = confidence_interval(data, confidence=0.99)
    return {
        "values": list(data),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci90": ci90,
        "ci95": ci95,
        "ci99": ci99,
    }


if __name__ == "__main__":
    test = [1, 2, 3, 4]
    print(get_metric_stat_dict(test))
