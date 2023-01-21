# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Polling detection module test fixtures"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

np.random.seed(10)


@pytest.fixture(scope="module")
def periodic_data():
    N = 86400
    start_ts = 1669852800
    end_ts = start_ts + N

    homo_pois = np.random.poisson(1.5, N)
    freq = 0.01666666666666
    periodic = (10 * np.sin(2 * np.pi * freq * np.arange(0, N))).astype("int")
    periodic[periodic < 0] = 0
    x = (periodic + homo_pois).astype("bool")
    ts = np.arange(start_ts, end_ts)[x]

    return ts


@pytest.fixture(scope="module")
def non_periodic_data():
    N = 86400
    start_ts = 1669852800
    end_ts = start_ts + N

    homo_pois = np.random.poisson(1.5, N)
    x = homo_pois.astype("bool")
    ts = np.arange(start_ts, end_ts)[x]

    return ts


@pytest.fixture(scope="module")
def tdds_timestamps():
    return np.array([
        1641038400, 1641038409, 1641038411, 1641038413, 1641038420, 1641038422,
        1641038425, 1641038446, 1641038449, 1641038453, 1641038454, 1641038466,
        1641038485
    ])


@pytest.fixture(scope="module")
def zeek_logs():
    return pd.read_csv(
        "tests/analysis/polling_detection/test_data/http-data.log",
        sep="\t",
        usecols=[
            "id.orig_h", "id.resp_h", "host", "id.resp_p", "method", "ts", "request_body_len"
        ]
    )