# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Polling detection module.

This module is used to highlight edges that are highly periodic and likely to be
generated automatically. The periodic edges could be software polling a server for
updates or malware beaconing and checking for instructions.

There is currently two techniques available for filtering polling data,
the class PeriodogramPollingDetector and TDDSPollingDetector.

The periodogram method requires only the connection times of events and scores
edges pased on how frequently they occur at the same time of day/week/year.

The TDDS (Time Delta Data Size) method requires the connection times and the data
size (for NetFlow data this could be the number of outbound bytes). Characteristics
of the time delta and data size distributions are combined to give an overall score.
"""
from collections import Counter
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from scipy import signal, special, stats

from ..common.utility import export


@export
class PeriodogramPollingDetector:
    """
    Polling detector using the Periodogram to detect strong frequencies.

    Methods
    -------
    detect_polling(timestamps, process_start, process_end, interval)
        Detect strong periodic frequencies

    """

    def __init__(self) -> None:
        """Create periodogram polling detector."""

    def _g_test(self, pxx: npt.NDArray, exclude_pi: bool) -> Tuple[float, float]:
        """
        Carry out fishers g test for periodicity.

        Fisher's g test tests the null hypothesis that the time series is gaussian white noise
        against the alternative that there is a deterministic periodic component[1]

        If the length of the time series is even then the intensity at pi should be excluded

        If the length of the power spectral density estimate is larger than 700 then an approximate
        p value is calculated otherwise the exact p value is calculate.

        This implementation was taken from the R package GeneCycle[2]

        Parameters
        ----------
        pxx: ArrayLike
            Estimate of the power spectral density

        exclude_pi: bool
            A bool to indicate whether the frequnecy located at pi should be removed.

        Returns
        -------
        Tuple[float, float]
            G test test statistic
            G test P value

        References
        ----------
        [1] M. Ahdesmaki, H. Lahdesmaki and O. Yli-Harja, "Robust Fisher's Test for Periodicity
        Detection in Noisy Biological Time Series," 2007 IEEE International Workshop on Genomic
        Signal Processing and Statistics, 2007, pp. 1-4, doi: 10.1109/GENSIPS.2007.4365817.
        [2] https://github.com/cran/GeneCycle/blob/master/R/fisher.g.test.R

        """
        if exclude_pi:
            pxx = pxx[:-1]

        pxx_length = len(pxx)
        test_statistic = np.max(pxx) / sum(pxx)
        upper = np.floor(1 / test_statistic).astype("int")

        if pxx_length > 700:
            p_value = 1 - (1 - np.exp(-pxx_length * test_statistic)) ** pxx_length
        else:
            compose = []
            for j in range(1, upper):
                compose.append(
                    (-1) ** (j - 1)
                    * np.exp(
                        np.log(special.binom(pxx_length, j))
                        + (pxx_length - 1) * np.log(1 - j * test_statistic)
                    )
                )

            p_value = sum(compose)

        p_value = min(p_value, 1)

        return test_statistic, p_value

    def detect_polling(
        self,
        timestamps: npt.NDArray,
        process_start: int,
        process_end: int,
        interval: int = 1,
    ) -> float:
        """
        Carry out periodogram polling detecton.

        Carries out the the procedure outlined in [1] to detect if the arrival times have a strong
        periodic component.
        The procedure estimates the periodogram for the data and passes the results to fishers G
        test.

        For more information run PeriodogramPollingDetector._g_test.__doc__

        This code was adapted from [2].

        Parameters
        ----------
        timestamps: ArrayLike
            An array like object containing connection arrival times as timestamps
        process_start: int
            The timestamp representing the start of the counting process
        process_end: int
            The timestamp representing the end of the counting process
        interval: int
            The interval in seconds between observations

        Returns
        -------
        p_val: float
            The p value from fishers G test

        References
        ----------
          [1] Heard, N. A. and Rubin-Delanchy, P. T. G. and Lawson, D. J. (2014) Filtering
          automated polling traffic in computer network flow data. In proceedings of IEEE
          Joint Intelligence and Security Informatics Conference 2014
          [2] https://github.com/fraspass/human_activity/blob/master/fourier.py

        """
        time_steps = np.arange(process_start, process_end, step=interval)
        counting_process = Counter(timestamps)

        dn_ = np.array([counting_process[t] for t in time_steps])
        dn_star = dn_ - len(timestamps) / len(time_steps)

        freq, pxx = signal.periodogram(dn_star)

        max_pxx_freq = freq[np.argmax(pxx)]

        print(
            (
                f"Dominant frequency detected at {round(1 / max_pxx_freq)} seconds\n"
                f"\tFrequency: {max_pxx_freq}\n"
                f"\tTime domain: {1 / max_pxx_freq}"
            )
        )

        if len(dn_star) % 2 == 0:
            _, p_val = self._g_test(pxx, True)
        else:
            _, p_val = self._g_test(pxx, False)

        return p_val


class TDDSPollingDetector:
    """
    Polling detector using time delta and data size distribution characteristics.

    Methods
    -------
    detect_polling(timestamps, data_sizes)
        Detect edges conducting polling behaviour

    """

    def __init__(self) -> None:
        """Create periodogram polling detector."""
    
    def _get_quantile(self, data: npt.NDArray, q: float) -> float:
        """
        Gets the desired quantile from an array

        The array is sorted prior to calculating the quantile

        Parameters
        ----------
        data: NDArray
            The data that the quantile should be calculated for
        q: float
            The quantile to calculate. Must be between 0 and 1
        
        Returns
        -------
        quantile: float
            The quantile requested
        
        """
        data = sorted(data)

        return np.quantile(data, q)

    def _bowleys_skewness(self, data: npt.NDArray) -> float:
        """
        Calculates Bowleys skewness measure

        Bowleys skewness is a quantile based method of estimating the skewness of a sample of data

        Parameters
        ----------
        data: NDArray
            Sample of data to calculate the skewness for

        Returns
        -------
        bowleys coefficient: float

        References
        ----------
        [1] Bowley, A. L. (1901). Elements of Statistics, P.S. King & Son, Laondon. Or in a
        later edition: BOWLEY, AL. "Elements of Statistics, 4th Edn
        (New York, Charles Scribner)."(1920).
        """
        data = sorted(data)

        q_1 = self._get_quantile(data, 0.25)
        q_2 = self._get_quantile(data, 0.5)
        q_3 = self._get_quantile(data, 0.75)

        if any([q_1 == q_2, q_1 == q_3, q_2 == q_3]):
            raise RuntimeError(
                "One or more quantiles are equal. Bowleys skew is unreliable if quantiles are equal"
            )
        
        return (q_3 + q_1 - 2 * q_2) / (q_3 - q_1)
        
    def _median_absolute_deviation(self, data: npt.NDArray) -> float:
        """
        Calculates the median absolute deviation (MAD) of a sample

        Parameters
        ----------
        data: NDArray
            The sample to calculate the MAD for

        Returns
        -------
        mean absolute deviation: float

        References
        ----------
        https://en.wikipedia.org/wiki/Median_absolute_deviation
        """
        absolute_deviation = np.abs(data - self._get_quantile(data, 0.5))

        return self._get_quantile(absolute_deviation, 0.5)

    def _connection_count_score(self, timestamps: npt.NDArray, jitter: float = 3600.0) -> float:
        """
        Calculates the number of connections per second

        The number of connections per second must be less than one

        Parameters
        ----------
        timestamps: NDArray
            The timestamps to calculate the connections per second

        Returns
        -------
        connections per second: float

        """
        conn_per_sec = len(timestamps) / ((max(timestamps) - min(timestamps)) / jitter)

        return min(1.0, conn_per_sec)

    def _smallness_score(self, data: npt.NDArray, scaling_factor: Union[int, float] = 65535.0) -> float:
        if scaling_factor == 0.0:
            val_err = ValueError(
                ("Value of 0 passed to scaling_factor argument. scaling_factor can be any value"
                "other than 0")
            )
            raise val_err

        return 1.0 - (stats.mode(data) / scaling_factor)
    
    def _skew_score(self, data: npt.NDArray) -> float:
        skew = self._bowleys_skewness(data)

        return 1.0 - np.abs(skew)

    def _madm_score(self, data: npt.NDArray) -> float:
        mid = self._get_quantile(data, 0.5)
        madm_score = 1.0
        if mid >= 1.0:
            madm_score = 1.0 - self._median_absolute_deviation(data) / mid
        
        return max(0, madm_score)
    
    def _ts_score(self, data: npt.NDArray) -> float:
        skew_score = self._skew_score(data)
        madm_score = self._madm_score(data)
        conn_score = self._connections_per_second(data)
        
        return np.ceil(
            (
                (
                    (skew_score + madm_score + conn_score) / 3.0
                ) * 1000
            ) / 1000
        )

    def _ds_score(self, data: npt.NDArray) -> float:
        skew_score = self._skew_score(data)
        madm_score = self._madm_score(data)
        smallness_score = self._smallness_score(data)
        
        return np.ceil(
            (
                (
                    (skew_score + madm_score + smallness_score) / 3.0
                ) * 1000
            ) / 1000
        )
    
    def _duration_score(self, timestamps: npt.NDArray, ts_start: int, ts_end: int) -> float:
        duration = np.ceil(
            ( ( (timestamps[-1] - timestamps[0]) / (ts_end - ts_start) ) * 1000 )  / 1000
        )

        return min(duration, 1.0)

    def _ts_hist_score(timestamps: npt.NDArray):
        hist, _ = np.histogram(timestamps)
        if len(hist) > 11:
            score1 = np.ceil(
                (4 / len(hist)) * 1000
            ) / 1000
            score1 = min(1.0, score1)
        
        score2 = np.ceil(np.std(timestamps))

    