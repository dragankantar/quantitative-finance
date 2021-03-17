import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime
import yfinance as yf

class returns_calc:

    def __init__(self, ticker, begin, end, initial_aum):
        self.ticker = None
        self.begin = None
        self.end = None
        self.ticker = ticker
        self.begin = begin
        self.end = end
        self.initial_aum = initial_aum
        self.company_name = None

        self.cal_days = None
        self.num_years = None
        self.shares_bought = None
        self.total_ret = None
        self.total_ret_aum = None
        self.pnl = None
        self.annualized_ror = None
        self.final_aum = None
        self.daily_ret = None
        self.current_aum = None
        self.avg_aum = None
        self.max_aum = None
        self.avg_ret_d = None
        self.sd_rets_d = None
        self.sharpe = None

        if self.ticker == None:
            sys.exit("Ticker not found!")
        elif self.begin == None and self.end == None:
            self.data = yf.download(self.ticker)
            self.data = self.data['Adj Close']
        elif self.begin == None:
            self.end_c = datetime.strptime(self.end,'%Y%m%d').date()
            self.data = yf.download(self.ticker, end=self.end_c)
            self.data = self.data['Adj Close']
        elif self.end == None:
            self.begin_c = datetime.strptime(self.begin,'%Y%m%d').date()
            self.data = yf.download(self.ticker, start=self.begin_c)
            self.data = self.data['Adj Close']
        else:
            self.begin_c = datetime.strptime(self.begin,'%Y%m%d').date()
            self.end_c = datetime.strptime(self.end,'%Y%m%d').date()
            self.data = yf.download(self.ticker, start=self.begin_c, end=self.end_c)
            self.data = self.data['Adj Close']

        self.begin_real = self.data.index[0].date()
        self.end_real= self.data.index[-1].date()

    def get_name(self):
        """Pulls the full name of the company from Yahoo Finance

        Returns:
            str: company name
        """
        self.company_name = yf.Ticker(self.ticker).info['longName']
        return self.company_name

    def calendar_days(self):
        """finds the number of calendar days between the first and the last day
            considered for calculations

        Returns:
            datetime.timedelta: number of calendar days between begin_real and end_real
        """
        self.cal_days = self.end_real-self.begin_real
        return self.cal_days.days

    def _num_years(self):
        """finds number of years between the begin_real and end_real

        Returns:
            float: number of years
        """
        self.cal_days = self.calendar_days()
        self.num_years = self.cal_days/356
        return self.num_years

    def total_stock_return(self):
        """calculates total net return of the stock

        Returns:
            float: total net return of the stock, expressed in given currency
        """
        self.shares_bought = self.initial_aum/self.data[0]
        self.total_ret = self.shares_bought*self.data[-1]-self.shares_bought*self.data[0]
        return self.total_ret

    def total_return_aum(self):
        """calculates profit and loss as a percentage of the initial assets under management

        Returns:
            float: profit and loss as percentage of the AuM (in percent)
        """
        self.pnl = self.pnl_calc()
        self.total_ret_aum = self.pnl/self.initial_aum
        return self.total_ret_aum

    def annualized_ror_calc(self):
        """calculates annualized rate of return of the asset

        Returns:
            float: annualized rate of return (in percent)
        """
        self.total_ret = self.total_stock_return()
        self.num_years = self._num_years()
        self.annualized_ror = ((self.initial_aum + self.total_ret)/self.initial_aum)**(1/self.num_years)-1
        return self.annualized_ror

    def final_aum_calc(self):
        """calculates assets under management at the end of the period (end_real)

        Returns:
            float: AuM at the end of the investment horizon
        """
        self.total_ret = self.total_stock_return()
        self.final_aum = self.initial_aum + self.initial_aum*self.total_ret
        return self.final_aum

    def _current_aum_calc(self):
        """calculates assets under management for every trading day

        Returns:
            numpy.ndarray: AuM for every trading day
        """
        self.daily_ret = self._daily_returns()
        self.current_aum = np.zeros(len(self.data))
        self.current_aum[0] = self.initial_aum
        for i in range(len(self.data)-1):
            self.current_aum[i+1] = self.current_aum[i] + self.daily_ret[i+1] * self.current_aum[i]
        return self.current_aum

    def avg_aum_calc(self):
        """calculates average daily (trading days) assets under managemend

        Returns:
            float: average daily AuM
        """
        self.current_aum = self._current_aum_calc()
        self.avg_aum = np.mean(self.current_aum)
        return self.avg_aum

    def max_aum_calc(self):
        """finds the highest level of assets under management in the investment period

        Returns:
            float: highest level of AuM in the period
        """
        self.current_aum = self._current_aum_calc()
        self.max_aum = max(self.current_aum)
        return self.max_aum

    def pnl_calc(self):
        """calculates profit and loss

        Returns:
            float: profit and loss
        """
        self.final_aum = self.final_aum_calc()
        self.pnl = self.final_aum-self.initial_aum
        return self.pnl

    def _daily_returns(self):
        """calculates daily returns (in percent)

        Returns:
            numpy.ndarray: daily returns (in percent)
        """
        self.daily_ret = self.data.pct_change()
        return self.daily_ret

    def avg_daily_ret(self):
        """finds the average daily return (in percent)

        Returns:
            float: average daily return (in percent)
        """
        self.avg_ret_d = self.data.pct_change().mean()
        return self.avg_ret_d

    def sd_portfolio_d(self):
        """calculates the standard deviation of the asset (daily data)

        Returns:
            float: standard deviation of the asset (daily data)
        """
        self.daily_ret = self._daily_returns()
        self.sd_rets_d = np.std(self.daily_ret)
        return self.sd_rets_d

    def sharpe_calc(self):
        """calculates Sharpe ratio of the asset in a given period

        Returns:
            float: Sharpe ratio of the asset
        """
        self.avg_ret_d = self.avg_daily_ret()
        self.sd_rets_d = self.sd_portfolio_d()
        self.sharpe = (100*self.avg_ret_d - 0.01)/self.sd_rets_d
        return self.sharpe

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
