import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf


class ReturnAnalytics:

    def __init__(self, ticker, period_begin, period_end):
        self.ticker = ticker
        self.period_begin = datetime.strptime(period_begin,'%Y%m%d').date()
        self.period_end = datetime.strptime(period_end,'%Y%m%d').date()

        self.data = yf.download(self.ticker, start=self.begin_c, end=self.end_c)
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