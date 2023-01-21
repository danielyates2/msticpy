# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Test polling detection module
"""
import numpy as np
import pandas as pd
import pytest

from msticpy.analysis import polling_detection as poll

__author__ = "Daniel Yates"


## ############# ##
## _get_quantile ##
## ############# ##

def test_get_quantile(tdds_timestamps):
    tdds = poll.TDDSPollingDetector()

    test_arr = sorted(np.diff(tdds_timestamps))

    assert tdds._get_quantile(test_arr, 0.5) == 3.5
    assert tdds._get_quantile(test_arr, 0.25) == 2.0
    assert tdds._get_quantile(test_arr, 0.75) == 9.75 


def test_get_quantile_unsorted(tdds_timestamps):
    tdds = poll.TDDSPollingDetector()

    test_arr = np.diff(tdds_timestamps)

    assert tdds._get_quantile(test_arr, 0.5) == 3.5
    assert tdds._get_quantile(test_arr, 0.25) == 2.0
    assert tdds._get_quantile(test_arr, 0.75) == 9.75 


## ################# ##
## _bowleys_skewness ##
## ################# ##

def test_bowleys_skewness(tdds_timestamps):
    tdds = poll.TDDSPollingDetector()

    test_arr = sorted(np.diff(tdds_timestamps))
    
    assert round(tdds._bowleys_skewness(test_arr), 5) == 0.61290


@pytest.mark.parametrize("transform", [np.array, pd.Series])
def test_bowleys_skewness_different_arrays(tdds_timestamps, transform):
    test_arr = transform(
        sorted(
            np.diff(tdds_timestamps)
            )
        )

    tdds = poll.TDDSPollingDetector()
    
    assert round(tdds._bowleys_skewness(test_arr), 5) == 0.61290


@pytest.mark.parametrize(
    "arr",
    [
        [1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,3,3,3], [1,1,1,3,3,3,3,3,3]
    ]
)
def test_bowleys_skewness_equal_quantiles(arr):
    test_arr = np.array(arr)

    tdds = poll.TDDSPollingDetector()

    with pytest.raises(RuntimeError):
        tdds._bowleys_skewness(test_arr)


## ########################## ##
## _median_absolute_deviation ##
## ########################## ##

def test_median_absolute_deviation(tdds_timestamps):
    tdds = poll.TDDSPollingDetector()

    test_arr = sorted(np.diff(tdds_timestamps))

    assert tdds._median_absolute_deviation(test_arr) == 2.0

## ####################### ##
## _connections_per_second ##
## ####################### ##

def test_connections_per_second(tdds_timestamps):
    tdds = poll.TDDSPollingDetector()

    assert tdds._connection_count_score(tdds_timestamps) == 1.0


def test_connections_per_second_result_less_than_one(tdds_timestamps):
    tdds = poll.TDDSPollingDetector()

    test_arr = tdds_timestamps.copy()
    test_arr[-1] = 1641034300

    assert round(tdds._connection_count_score(test_arr), 5) == 0.28084

## ################ ##
## _smallness_score ##
## ################ ##

def test_smallness_score(tdds_timestamps):
    tdds = poll.TDDSPollingDetector()

    test_arr = sorted(np.diff(tdds_timestamps))
    median = tdds._get_quantile(test_arr, 0.5)

    assert round(tdds._smallness_score(median, 8192.0), 5) == 0.99957

def test_smallness_score_zero_jitter(tdds_timestamps):
    tdds = poll.TDDSPollingDetector()

    test_arr = sorted(np.diff(tdds_timestamps))
    median = tdds._get_quantile(test_arr, 0.5)

    with pytest.raises(ValueError):
        tdds._smallness_score(median, 0.0)

## ########### ##
## Integration ##
## ########### ##

def test_polling_detected(zeek_logs):
    tdds = poll.TDDSPollingDetector()

    zeek_logs["edges"] = (
        zeek_logs["id.orig_h"] + "::" +
        zeek_logs["id.resp_h"] + "::" +
        zeek_logs["host"] + "::" +
        zeek_logs["id.resp_p"].astype("str") + "::" +
        zeek_logs["method"]
    )

    zeek_logs_scored = zeek_logs.groupby("edges").apply(lambda x: tdds.detect_polling(x["ts"], x["request_body_len"]))
    expected_scores = pd.read_csv("test_data/expected_scores.csv")

    assert zeek_logs_scored == expected_scores

    