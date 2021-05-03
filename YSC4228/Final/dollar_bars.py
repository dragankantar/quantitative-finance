import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# TODO: add docstrings
# TODO: try doing it without a for loop
# TODO: remove unused vars

class Sampler():
    def __init__(self, raw_data, dollar_size):
        self.raw_data = raw_data
        self.dollar_size = dollar_size

    def get_data(self):
        data = pd.read_csv(self.raw_data, index_col='Time', parse_dates=['Time']).dropna()
        return data

    #def _get_group_values(self, groups):

    def get_dollar_bars(self, data):
        #data = data.copy(deep=True)
        data['dollar_volume'] = data['Volume'] * data['VWAP']

        data['cumulative_dollar_vol'] = data['dollar_volume'].cumsum()
        cum_idx  = (data['cumulative_dollar_vol'] / self.dollar_size ).round(0).astype(int).values
        groups = data.groupby(cum_idx)

        bar_series = pd.DataFrame()

        bar_series['Open'] = groups['Open'].first()
        bar_series['Close'] = groups['Close'].last()

        bar_series['Volume'] = groups['Volume'].sum()
        bar_series['Ticks'] = groups['Ticks'].sum()
        bar_series['dollar_volume'] = groups['dollar_volume'].sum()

        bar_series['VWAP'] = bar_series['dollar_volume']/bar_series['Volume']

        bar_series = bar_series.set_index(groups.apply(lambda x: x.index[0]))

        return bar_series

    def resample_weekly(self, data):
        resampled_data = data.resample('1W').count()['Close']
        resampled_data = pd.DataFrame(resampled_data, columns = ['Close'])
        resampled_data.index = pd.to_datetime(resampled_data.index)

        return resampled_data

    def plot_one(self, dollar_bars):
        dollar_bars['Close'].plot(figsize=(12,5), title='Dollar Bars')
        plt.show()

    def plot_two(self, original_bars_weekly, dollar_bars_weekly):
        ax = plt.gca()
        original_bars_weekly.plot(figsize=(12,5), y='Original Bars', ax=ax, legend=True)
        dollar_bars_weekly.plot(figsize=(12,5), y='Dollar Bars', ax=ax, legend=True)
        ax.legend(['Original Bars', 'Dollar Bars'])
        #, y='Original Bars'
        #, y='Dollar Bars'
        plt.show()

    def jarque_bera_test(self, dataframe):
        close = dataframe["Close"].pct_change()[1:]
        test = stats.jarque_bera(close)

        return test.statistic
