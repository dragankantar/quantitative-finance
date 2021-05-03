"""
This script defines Sampler class of objects.

Usage: No need to run directly. Just run get_dollar_bars.py through a shell.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


class Sampler():

    def __init__(self, filename, dollar_size):
        self.filename = filename
        self.dollar_size = dollar_size

    def get_data(self):
        """Loads the raw time series data

        Returns:
            pandas.DataFrame: Pandas dataframe of a .csv data file
        """
        data = pd.read_csv(self.filename, index_col='Time', parse_dates=['Time']).dropna()

        return data

    def get_dollar_bars(self, data):
        """Gets dollar bars from raw time series data

        Args:
            data (pandas.DataFrame): Pandas dataframe of raw time series data

        Returns:
            pandas.DataFrame: Dataframe with dollar bar time series
        """
        data = data.copy(deep=True)
        data['Dollar Volume'] = data['Volume'] * data['VWAP']

        data['Cumulative Dollar Volume'] = data['Dollar Volume'].cumsum()
        cumulative_index = (data['Cumulative Dollar Volume'] / self.dollar_size ).round(0).astype(int).values
        groups = data.groupby(cumulative_index)

        bar_series = pd.DataFrame()
        bar_series['Open'] = groups['Open'].first()
        bar_series['Close'] = groups['Close'].last()
        for i in ['Volume', 'Ticks', 'Dollar Volume']:
            bar_series[i] = groups[i].sum()
        bar_series['VWAP'] = bar_series['Dollar Volume']/bar_series['Volume']
        bar_series = bar_series.set_index(groups.apply(lambda x: x.index[0]))

        return bar_series

    def resample_weekly(self, data):
        """Resamples data into weekly bars

        Args:
            data (pandas.DataFrame): Pandas dataframe with time series data to be resampled

        Returns:
            pandas.DataFrame: Weekly resampled time series data
        """
        resampled_data = data.resample('1W').count()['Close']
        resampled_data = pd.DataFrame(resampled_data, columns = ['Close'])
        resampled_data.index = pd.to_datetime(resampled_data.index)

        return resampled_data

    def plot_one(self, dollar_bars):
        """Plot dollar bar series

        Args:
            dollar_bars (pandas.DataFrame): Dollar bars series data
        """
        dollar_bars['Close'].plot(figsize=(11,5.5), ylabel='Close Price', xlabel='Time', title='Dollar Bars')
        plt.show()

    def plot_two(self, original_bars_weekly, dollar_bars_weekly):
        """Plots time series of the counts of the original and the dollar bars weekly samples

        Args:
            original_bars_weekly (pandas.DataFrame): Dataframe with original data weekly sampled
            dollar_bars_weekly (pandas.DataFrame): Dataframe with dollar bars data weekly sampled
        """
        ax = plt.gca()
        original_bars_weekly.plot(figsize=(11,5.5), ylabel='Close Price', xlabel='Time', ax=ax, legend=True)
        dollar_bars_weekly.plot(figsize=(11,5.5), ylabel='Close Price', xlabel='Time', ax=ax, legend=True)
        ax.legend(['Original Bars', 'Dollar Bars'])
        plt.show()

    def jarque_bera_test(self, dataframe):
        """Performs Jarque-Bera Normality Test on the given data

        Args:
            dataframe (pandas.DataFrame): Data on which to perform Jarque-Bera Normality Test

        Returns:
            float: Test statistic
        """
        close = dataframe["Close"].pct_change()[1:]
        test = stats.jarque_bera(close)

        return test.statistic
