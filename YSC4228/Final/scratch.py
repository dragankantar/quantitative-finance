#get_dollar_bars.py --file e_mini_sp_series.csv --dollar_size 18000000
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# TODO: test
# TODO: jb test

# TODO: plot 1
# TODO: plot 2

def enable_parsing(parser):
    """
    Function that adds the arguments to our parser

    Returns: None
    """
    parser.add_argument("--file", help="Time series data", required=True)
    parser.add_argument("--dollar_size", help="Dollar bar size", required=True)

    return parser

parser = argparse.ArgumentParser()
parser = enable_parsing(parser)
args = parser.parse_args()

class Bar():
    def __init__(self, file, dollar_size):
        self.file = file
        self.dollar_size = dollar_size

    def data(self):
        data = pd.read_csv(self.file, index_col='Time', parse_dates=['Time']).dropna()
        return data

    def jarque_bera_test(self, dataframe):
        close = dataframe["Close"].pct_change()[1:]
        test = stats.jarque_bera(close)
        return test.statistic, test.pvalue



    def plot(self):
        """plots the assets under management over time
        """
        self.current_aum = self._current_aum_calc()
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.data.index, self.current_aum)
        ax.set_xlabel('Time')
        ax.set_ylabel('AUM')
        plt.show()

import os
os.chdir("C:/Users/draga/Documents/Finance/quantitative-finance/YSC4228/Final")

df = pd.read_csv("e_mini_sp_series.csv")
df.index = pd.to_datetime(df.index)
df.count()["Close"]
df.resample('1W').count()["Close"]
ws = df.resample('1W').count()['Close']
ws = pd.DataFrame(ws, columns = ['Close'])
ws


###
import os
os.chdir("C:/Users/draga/Documents/Finance/quantitative-finance/YSC4228/Final")

dollar_size = 18000000
data = pd.read_csv("e_mini_sp_series.csv", index_col='Time', parse_dates=['Time']).dropna()

data = data.copy(deep=True)
data['dollar_volume'] = data['Volume'] * data['VWAP']

data['cumulative_dollar_vol'] = data['dollar_volume'].cumsum()
cum_idx = (data['cumulative_dollar_vol'] / dollar_size ).round(0).astype(int).values
groups = data.groupby(cum_idx)

bar_series = pd.DataFrame()

bar_series['Open'] = groups['Open'].first()
bar_series['Close'] = groups['Close'].last()

bar_series['Volume'] = groups['Volume'].sum()
bar_series['Ticks'] = groups['Ticks'].sum()
bar_series['dollar_volume'] = groups['dollar_volume'].sum()

bar_series['VWAP'] = bar_series['dollar_volume']/bar_series['Volume']
bar_series.drop('dollar_volume', axis=1, inplace=True)

bar_series = bar_series.set_index(groups.apply(lambda x: x.index[0]))

bar_series


for field in [volume, tick count, dollar_volume]:
    bar_Series[field] = groups[field].sum()
data
cum_idx
groups
###


    def _get_bars(self, series, bar_size, bar_col):
        series = series.copy(deep=True)
        dict = self._col_mapping

        series[dict[self.DOL_VOLUME]] = series[dict[self.VOLUME]] * series[dict[self.VWAP]]

        cum_col = 'cum_' + bar_col
        series[cum_col] = series[dict[bar_col]].cumsum()
        cum_idx = (series[cum_col] / bar_size).round(0).astype(int).values
        groups = series.groupby(cum_idx)

        bar_series = self._get_group_values(groups)
        return bar_series

#

def get_dollar_bars(self, data):
    data['Dollar Volume'] = data['VWAP'] * data['Volume']
    data["Time"] = data.index
    data = data.set_index(pd.Index(range(len(data))))

    bars = np.zeros((len(data), 2))
    bars_counter = 0
    current_dollars = 0

    for i in range(len(data)):
        current_dollars += data['Dollar Volume'][i]
        if current_dollars >= self.dollar_size:
            bars[bars_counter][0] = data.index[i]
            bars[bars_counter][1] = data['Close'][i]
            bars_counter = bars_counter + 1
            current_dollars = 0

    dollar_bars = pd.DataFrame(bars[:bars_counter], columns = ['Time', 'Close'])
    dollar_bars['Time'] = pd.to_datetime(data['Time'])
    dollar_bars = dollar_bars.set_index('Time')

    return dollar_bars


# plot 1





# plot 2







#JB test

close = dataframe["Close"].pct_change()[1:]
test = stats.jarque_bera(close)

return test.statistic

df1 = DV one
df2 = original returns


    def _get_group_values(self, groups):


    def _get_bars(self, series, bar_size, bar_col):
        series = series.copy(deep=True)
        dict = self._col_mapping

        series[dict[self.DOL_VOLUME]] = series[dict[self.VOLUME]] * series[dict[self.VWAP]]

        cum_col = 'cum_' + bar_col
        series[cum_col] = series[dict[bar_col]].cumsum()
        cum_idx = (series[cum_col] / bar_size).round(0).astype(int).values
        groups = series.groupby(cum_idx)

        bar_series = self._get_group_values(groups)
        return bar_series

    def working_dollar_bars(series, dollar_size):
        series = series.copy(deep=True)
        series['dollar_volume'] = series['Volume'] * series['VWAP']

        series['cumulative_dollar_vol'] = series['dollar_volume'].cumsum()
        cum_idx  = (series['cumulative_dollar_vol'] / dollar_size ).round(0).astype(int).values

 



    def get_dollar_bars_nofor(self, series, dollar_size, self.DOL_VOLUME):
        return self._get_bars(self, series, dollar_size, self.DOL_VOLUME)


