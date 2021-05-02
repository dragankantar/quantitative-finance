#get_dollar_bars.py --file e_mini_sp_series.csv –-dollar_size 18000000
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def enable_parsing(parser):
    """
    Function that adds the arguments to our parser

    Returns: None
    """
    parser.add_argument("--file", help="Time series data", required=True)
    parser.add_argument("--dollar_size", help="Dollar size amount", required=True)

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
