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

The TDDS method is a port of the RITA beaconing detection from v4.8.0.
"""
from collections import Counter, OrderedDict
from typing import Tuple, Union, Dict, List

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

    This implementation is a translation from the RITA package written by active countermeasures.
    Repo: https://github.com/activecm/rita
    Beaconing detector: https://github.com/activecm/rita/blob/master/pkg/beacon/analyzer.go

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
        Calculates the median absolute deviation (MAD) of a sample of data

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

    def _truncate_decimal(self, value: float, precision: int = 3) -> float:
        """
        Truncates a decimal to the specified precision.
        e.g _truncate_decimal(0.123456, 3) -> 0.123

        Parameters
        ----------
        value: float
            The number that needs to be truncated

        Returns
        -------
        truncated decimal: float
        """
        return np.ceil(value * 10**precision) / 10**precision

    def _smallness_score(self, bytes_list: npt.NDArray, scaling_factor: Union[int, float] = 65535.0) -> float:
        """
        Calculates a score for data sizes, smaller data sizes receive a higher score.
        The scaling_factor determines how sensitive the score is

        e.g a bytes list with a mode of 2 and a scaling factor of 4 will have
        a smallness score of 0.5. If the scaling factor is increased to 32 the smallness score
        will be 0.9375

        Parameters
        ----------
        bytes_list: NDArray
            Array containing byte sizes
        
        scaling_factor: Union[int, float]
            How sensitive to make the scoring. Larger scaling factors will have a greater impact
            on the score.
        
        Returns
        -------
        smallness score: float
        """
        if scaling_factor == 0.0:
            raise ValueError(
                ("Value of 0 passed to scaling_factor argument. scaling_factor can be any value"
                "other than 0")
            )

        mode = stats.mode(bytes_list).mode[0]

        return 1.0 - (mode / scaling_factor)
    
    def _skew_score(self, data: npt.NDArray) -> float:
        """
        Calculates a skewness score for an array of delta times or an array of byte sizes.

        Parameters
        ----------
        data: NDArray
            Either a delta time array or byte size distribution

        Returns
        -------
        skewness score: float
        """
        skew = self._bowleys_skewness(data)

        return 1.0 - np.abs(skew)

    def _madm_score(self, data: npt.NDArray) -> float:
        """
        Calculates a median absolute deviation score.

        Parameters
        ----------
        data: NDArray
            Either a delta time array or array of byte sizes

        Returns
        -------
        skewness score: float
        """
        mid = self._get_quantile(data, 0.5)
        madm_score = 1.0
        if mid >= 1.0:
            madm_score = 1.0 - self._median_absolute_deviation(data) / mid
        
        return max(0, madm_score)
    
    def _ts_score(self, delta_times: npt.NDArray) -> float:
        """
        Calculates the final ts score

        Parameters
        ----------
        delta_times: NDArray
            Array of timestamp deltas
        
        Returns
        -------
        ts score: float
        """
        skew_score = self._skew_score(delta_times)
        madm_score = self._madm_score(delta_times)
        
        return self._truncate_decimal((skew_score + madm_score) / 2.0)
    
    def _ds_score(self, bytes_list: npt.NDArray) -> float:
        """
        Calculates the final ds score

        Parameters
        ----------
        bytes_list: NDArray
            Array containing bytes sizes

        Returns
        -------
        ds score: float
        """
        skew_score = self._skew_score(bytes_list)
        madm_score = self._madm_score(bytes_list)
        smallness_score = self._smallness_score(bytes_list)
        
        return self._truncate_decimal((skew_score + madm_score + smallness_score) / 3.0)
    
    def _duration_score(
            self,
            timestamps: npt.NDArray,
            minimum: int, 
            maximum: int,
            total_bars: int,
            longest_run: int,
            min_hours_seen: int = 6,
            consistency_ideal_hours_seen: int = 12
        ) -> float:
        """
        Calculates the duration score

        Parameters
        ----------
        timestamps: NDArray
            Array containing timestamps
        
        minimum: int
            Minimum timestamp in the whole dataset

        maximum: int
            Maximum timestamp in the whole dataset
        
        total_bars: int
            Number of hours represented in the connection frequency histogram
        
        min_hours_seen: int
            The threshold value of the minimum number of hours to be present in the
            connection frequency histogram.

        consistency_ideal_hours_seen: int
            Ideal number of consecutive hours for the consistency score

        Returns
        -------
        duration score: float
        """
        if total_bars > min_hours_seen:
            coverage_score = self._truncate_decimal((timestamps[-1] - timestamps[0]) / (maximum - minimum))
            coverage_score = min(1.0, coverage_score)

            consistency_score = self._truncate_decimal(longest_run / consistency_ideal_hours_seen)
            consistency_score = min(1.0, consistency_score)

        return max(coverage_score, consistency_score)
    
    def _create_buckets(self, minimum: int, maximum: int, size: int = 24) -> List[int]:
        """
        Creates the buckets for the histogram

        Parameters
        ----------
        minimum: int
            Minimum value for the histogram

        maximum: int
            Maximum value for the histogram

        size: int
            The number of bins in the histogram
        
        Returns
        -------
        bucket_divs: List[int]
            The bucket divisions for the histogram
        """
        total = size + 1
        step = (maximum - minimum) / (total - 1)
        step = np.floor(step)

        bucket_divs = [minimum + (i * step) for i in range(1, total)]
        bucket_divs = [minimum] + bucket_divs

        bucket_divs[total - 1] = maximum

        return bucket_divs
    
    def _create_histogram(self, timestamps: npt.NDArray, bucket_divs: List[int], bimodal_bucket_size: float) -> Tuple[npt.NDArray, OrderedDict, int, int, int]:
        """
        Creates the histogram and calculates statistics associated with the histogram

        Parameters
        ----------
        timestamps: NDArray
            Array containing timestamps

        bucket_divs: List[int]
            The bucket divisions for the histogram
        
        bimodal_bucket_size: float
            Determines how forgiving the bimodal analysis is to variation

        Returns
        -------
        freq: NDArray
            The values of the histogram

        freq_count:
            Count of the number of histogram bars in each bucket for the bimodal analysis  
        
        total:
            The total number of timestamps

        total_bars:
            The total number of non zero bars in the histogram

        longest_run:
            The longest number of consecutive hours observed in the connection frequency histogram
        """
        freq, _ = np.histogram(timestamps, bins=bucket_divs)

        longest_run = 0
        current_run = 0
        for i in np.concatenate((freq, freq)):
            if i > 0:
                current_run += 1
            else:
                if current_run > longest_run:
                    longest_run = current_run

                current_run = 0
        
        if current_run > longest_run:
            longest_run = current_run
        
        total = sum(freq)

        total_bars = len(freq[freq != 0])

        bucket_size = np.ceil(np.max(freq) * bimodal_bucket_size)
        bucket = (np.floor(freq / bucket_size) * bucket_size).astype("int64")

        bucket = bucket[bucket != 0]

        freq_count = OrderedDict(sorted(Counter(bucket).items()))

        return freq, freq_count, total, total_bars, longest_run

    def _ts_hist_score(self, timestamps: npt.NDArray, minimum: int, maximum: int, bimodal_bucket_size: float = 0.05) -> Tuple[float, int, int]:
        """
        Calculate the connection frequency histogram score

        Parameters
        ----------
        timestamps: NDArray
            Array containing timestamps

        minimum: int
            Minimum value for the histogram

        maximum: int
            Maximum value for the histogram

        bimodal_bucket_size: float
            Determines how forgiving the bimodal analysis is to variation

        Returns
        -------
        ts_hist_score: float
            The largest score out of the cv_score and bimodal_fit_score
        
        total_bars:
            The total number of non zero bars in the histogram

        longest_run:
            The longest number of consecutive hours observed in the connection frequency histogram
        """
        bucket_divs = self._create_buckets(minimum, maximum)

        freq, freq_count, total, total_bars, longest_run = self._create_histogram(timestamps, bucket_divs, bimodal_bucket_size)

        freq_mean = np.mean(freq)
        freq_sd = np.std(freq)

        cv = freq_sd / freq_mean
        cv = min(1.0, cv)

        cv_score = self._truncate_decimal(1.0 - cv)
        cv_score = min(1, cv_score)

        bimodal_min_hours_seen = 11
        bimodal_outlier_removal = 1
        if total_bars >= bimodal_min_hours_seen:
            largest = 0
            second_largest = 0
            for value in freq_count.values():
                if value > largest:
                    second_largest = largest
                    largest = value
                elif value > second_largest:
                    second_largest = value

            bimodal_fit = (largest + second_largest) / max(total_bars - bimodal_outlier_removal, 1)

        bimodal_fit_score = self._truncate_decimal(bimodal_fit)
        bimodal_fit_score = min(bimodal_fit_score, 1.0)

        return max(cv_score, bimodal_fit_score), total_bars, longest_run

    def detect_polling(self, timestamps: npt.NDArray, bytes_list: npt.NDArray, weights: Dict[str, float]) -> float:
        """
        Applies the RITA beaconing detectioning algorithm developed by ActiveCountermeasures[1].

        The method consists of 4 scores:

            1. ts_score: 
            2. ds_score:
            3. hist_score: 
            4. duration_score: 
        
        The output is a weighted combination of these 4 scores.

        Parameters
        ----------
        timestamps: NDArray
            Array containing timestamps
        
        bytes_list: NDArray
            Array containing bytes sizes
        
        weights: Dict[str, float]
            Dictionary of weights to apply to each score. Must sum to 1

        Returns
        -------
        polling_score: float
            Final score for the timestamps and bytes_list. The higher the score, the more likely it
            is to be a beacon
        
        References
        ----------
        [1] https://github.com/activecm/rita/blob/master/pkg/beacon/analyzer.go
        """
        minimum = np.min(timestamps)
        maximum = np.max(timestamps)

        ts_score = self._ts_score(timestamps)
        ds_score = self._ds_score(bytes_list)
        hist_score, total_bars, longest_run = self._ts_hist_score(timestamps, minimum, maximum)
        duration_score = self._duration_score(timestamps, minimum, maximum, total_bars, longest_run)

        return self._truncate_decimal(
            (ts_score * weights["ts_weight"]) +
            (ds_score * weights["ds_weight"]) +
            (duration_score * weights["duration_weight"]) +
            (hist_score * weights["hist_weight"])
        )
