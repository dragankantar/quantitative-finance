import numpy as np
import pandas as pd
import yfinance as yf


class ReturnAnalytics:

    def __init__(self, ticker, period_begin, period_end):
        self.ticker = ticker
        self.period_begin = period_begin
        self.period_end = period_end
