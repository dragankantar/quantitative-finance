'''
Pytest to test the sampler.

Usage: Run "pytest" in a shell in the directory containing this file.
'''
import pytest
import pandas as pd
from dollar_bars import Sampler


def test_time_series():
    data = pd.read_csv("e_mini_sp_series.csv", index_col='Time', parse_dates=['Time'])
    dollar_size = 600000

    subsample = data[-6:]
    assert list(subsample.Volume) == [176, 94, 150, 78, 174, 244]
    assert list(subsample.VWAP) == [1992.8125, 1992.734042553192, 1992.515, 1992.5, 1992.433908045977, 1992.506147540984]

    sampler = Sampler(data, dollar_size)
    subsample_dollar_bars = sampler.get_dollar_bars(subsample)
    assert list(subsample_dollar_bars.Volume) == [420, 252, 244]
    assert list(subsample_dollar_bars.VWAP) == [1992.6886904761905, 1992.454365079365, 1992.506147540984]
