import numpy as np
import matplotlib.pyplot as plt
import statistics
import datetime
import pandas as pd
from yahoofinancials import YahooFinancials
from dateutil.relativedelta import relativedelta
import pandas_datareader as web
import yfinance as yf
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt


class BackTester():

    def __init__(self, tickers):
        self.tickers = tickers

    def get_dates(self, test_months, backtest_months):
        end_date = datetime.date.today()
        start_date = end_date - relativedelta(months=test_months)
        back_test_date = end_date - relativedelta(months=backtest_months)
        return (start_date, end_date, back_test_date)

    def get_data_frame(self, start_date, end_date):
        d = yf.download(self.tickers, start=start_date, end=end_date)
        d = d["Adj Close"].dropna(how="all")
        return d

    def get_realised_stats(self, optimized_portfolio, df):
        return None

    def get_portfolio(self, df, optimizer):
       
        if optimizer == "hrp":
            returns = df.pct_change().dropna()
            hrp = HRPOpt(returns)
            weights = hrp.optimize()
            performance = hrp.portfolio_performance(verbose=True)
            return weights, performance
        else:
            mu = mean_historical_return(df)
            S = CovarianceShrinkage(df).ledoit_wolf()
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe() if optimizer == "msr" else ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance()
            return cleaned_weights, performance
            

        
        

           
    
