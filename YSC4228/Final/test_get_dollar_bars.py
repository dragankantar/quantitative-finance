import pytest
import pandas as pd
from dollar_bars import Sampler

def test():
    tick_series = pd.read_csv("e_mini_sp_series.csv", index_col='Time', parse_dates=['Time'])

    small_tick_series = tick_series[-6:] # TODO: rename this
    assert list(small_tick_series.Volume) == [176, 94, 150, 78, 174, 244]
    assert list(small_tick_series.VWAP) == [1992.8125, 1992.734043, 1992.515, 1992.5, 1992.433908, 1992.506148]

    sampler = Sampler(tick_series)
    dollar_bars = sampler.get_dollar_bars_(small_tick_series, dollar_size=600000)

    assert list(dollar_bars.Volume) == [420, 252, 244]
    assert list(dollar_bars.VWAP) == [1992.6886904761905, 1992.454365079365, 1992.506147540984]
