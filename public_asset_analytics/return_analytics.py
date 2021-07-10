import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf


class ReturnAnalytics:

    def __init__(self, ticker, period_begin, period_end):
        self.ticker = ticker
        self.period_begin = datetime.strptime(period_begin,'%Y%m%d').date()
        self.period_end = datetime.strptime(period_end,'%Y%m%d').date()

        self.data = yf.download(self.ticker, start=self.period_begin, end=self.period_end)
        self.data = self.data['Adj Close']

    def get_name(self):
        """Pulls the full name of the company from Yahoo Finance

        Returns:
            str: company name
        """
        self.company_name = yf.Ticker(self.ticker).info['longName']
        return self.company_name

    def total_asset_return(self):
        """calculates total net return of the asset

        Returns:
            float: total net return of the asset, as expressed in given currency
        """
        self.total_return = (self.data[-1]-self.data[0])/self.data[0]
        return self.total_return

    def daily_returns(self):
        """calculates daily returns (in percent)

        Returns:
            numpy.ndarray: daily returns (in percent)
        """
        daily_returns = self.data.pct_change()
        return daily_returns

    def average_daily_returns(self):
        """finds the average daily return (in percent)

        Returns:
            float: average daily return (in percent)
        """
        average_daily_returns = self.data.pct_change().mean()
        return average_daily_returns

    def standard_deviation(self):
        """calculates the standard deviation of the asset (daily data)

        Returns:
            float: standard deviation of the asset (daily data)
        """
        standard_deviation = np.std(self.data.pct_change())
        return standard_deviation

    def sharpe_ratio(self, risk_free_rate):
        sharpe_ratio = (100*self.data.pct_change().mean()-risk_free_rate)/np.std(self.data.pct_change())
        return sharpe_ratio

    def plot_daily_returns(self):
        daily_returns = self.data.pct_change()

        fig, ax = plt.subplots()
        ax.plot(daily_returns)
        ax.set_ylabel('%s' % self.ticker)
        ax.set_xlabel('Time')
        fig.suptitle('Daily Returns')
        fig.show()
