import numpy as np
import matplotlib.pyplot as plt
import statistics
import datetime
import pandas as pd
from yahoofinancials import YahooFinancials
from dateutil.relativedelta import relativedelta
import pandas_datareader as web
import yfinance as yf


class BackTester():

    def __init__(self, tickers):
        self.tickers = tickers
       

    def get_dates(self, test_months):
        end_date = datetime.date.today()
        start_date = end_date - relativedelta(months=test_months)
        return (start_date, end_date)

    def get_data_frame(self, start_date, end_date):
        stocks = self.tickers
        d = web.DataReader(stocks, 'yahoo', start_date, end_date)
        return d[["Adj Close"]]

