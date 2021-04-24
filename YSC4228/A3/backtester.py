import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
import yfinance as yf
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt

# TODO: add docstrings
# ?: How to do purging (and embargoing)

class BackTester():

    def __init__(self, tickers):
        self.tickers = tickers

    def get_dates(self, test_months, backtest_months):
        """Calculate the dates for train and test splits for Walk-Forward methos

        Args:
            test_months (int): number of months used for testing
            backtest_months (int): number of months used for training

        Returns:
            datetime.date: dates at the start and end of training and testing period
        """
        end_date = datetime.date.today()
        start_date = end_date - relativedelta(months=test_months)
        back_test_date = end_date - relativedelta(months=backtest_months)
        return start_date, end_date, back_test_date

    def get_data(self, back_test_date, start_date, end_date):
        """Downloads the share price data

        Args:
            back_test_date (datetime.date): beginning of the training period
            start_date (datetime.date): end of the training period and beginning of the test period
            end_date (datetime.date): end of the test period (today's date)

        Returns:
            [pandas.Series]: daily share price data
        """
        train = yf.download(self.tickers, start=back_test_date, end=start_date)
        train = train["Adj Close"].dropna(how="all")

        test = yf.download(self.tickers, start=start_date, end=end_date)
        test = test["Adj Close"].dropna(how="all")

        data = yf.download(self.tickers, start=back_test_date, end=end_date)
        data = data["Adj Close"].dropna(how="all")

        return train, test, data

    def wf(self, train, optimizer, test, test_months, annual_risk_free_rate=0.02):
        """Walk-Forward backtesting method

        Args:
            train (pandas.Series): training data
            optimizer (str): portfolio optimizer for PyPortfolioOpt
            test (pandas.Series): test data
            test_months (int): number of testing months
            annual_risk_free_rate (float, optional): annual risk free rate used in calculating Sharpe ratio. Defaults to 0.02.

        Returns:
            [pandas.Series, float]: expected and realised asset performance
        """
        if optimizer == "hrp":
            returns = train.pct_change().dropna()
            hrp = HRPOpt(returns)
            weights = hrp.optimize()
            weights = pd.Series(weights)
            performance = hrp.portfolio_performance(verbose=True)

            realised_annual_return = sum(weights*((test.iloc[-1]/test.iloc[0])**(12/test_months)-1))
            realised_annual_volatility = sum(weights*np.std(test))*np.sqrt(12)
            realised_sharpe_ratio = (realised_annual_return-annual_risk_free_rate)/realised_annual_volatility

            return weights, performance, realised_annual_return, realised_annual_volatility, realised_sharpe_ratio
        else:
            mu = mean_historical_return(train)
            S = CovarianceShrinkage(train).ledoit_wolf()
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe() if optimizer == "msr" else ef.min_volatility()
            weights = pd.Series(weights)
            performance = ef.portfolio_performance()

            realised_annual_return = sum(weights*((test.iloc[-1]/test.iloc[0])**(12/test_months)-1))
            realised_annual_volatility = sum(weights*np.std(test))*np.sqrt(12)
            realised_sharpe_ratio = (realised_annual_return-annual_risk_free_rate)/realised_annual_volatility

            return weights, performance, realised_annual_return, realised_annual_volatility, realised_sharpe_ratio

    def cv(self, back_test_months, data, optimizer, test_months, annual_risk_free_rate=0.02):
        """Cross-Validation backtesting method

        Args:
            back_test_months (int): number of backtesting months
            data (pandas.Series): data that includes both training and testing data
            optimizer (str): portfolio optimizer for PyPortfolioOpt
            test_months (int): number of testing months
            annual_risk_free_rate (float, optional): annual risk free rate used in calculating Sharpe ratio. Defaults to 0.02.

        Returns:
            [pandas.Series, float]: expected and realised asset performance
        """
        embargo = np.round_(0.01*len(data), decimals=0)
        all_weights = np.zeros((back_test_months, np.shape(data)[1]))
        all_realised_annual_return = np.zeros(back_test_months)
        all_realised_annual_volatility = np.zeros(back_test_months)
        all_realised_sharpe_ratio = np.zeros(back_test_months)

        for i in range(back_test_months):

            test_start = i*len(data)/back_test_months
            test_end = test_start+len(data)/back_test_months-1
            test = data.iloc[test_start:test_end, :]
            train = data.iloc[np.r_[0:test_start, test_end+embargo:-1]]

            if optimizer == "hrp":
                train_returns = train.pct_change().dropna()
                hrp = HRPOpt(train_returns)
                weights = hrp.optimize()
                weights = pd.Series(weights)
                all_weights[i] = weights
                performance = hrp.portfolio_performance(verbose=True)
            else:
                mu = mean_historical_return(train)
                S = CovarianceShrinkage(train).ledoit_wolf()
                ef = EfficientFrontier(mu, S)
                weights = ef.max_sharpe() if optimizer == "msr" else ef.min_volatility()
                weights = pd.Series(weights)
                all_weights[i] = weights
                performance = ef.portfolio_performance()

            all_realised_annual_return[i] = sum(weights[i]*((test.iloc[-1]/test.iloc[0])**(12/test_months)-1))
            all_realised_annual_volatility[i] = sum(weights[i]*np.std(test))*np.sqrt(12)
            all_realised_sharpe_ratio = (all_realised_annual_return[i]-annual_risk_free_rate)/all_realised_annual_volatility[i]

        weights = np.mean(all_weights)
        realised_annual_return = np.mean(all_realised_annual_return)
        realised_annual_volatility = np.mean(all_realised_annual_volatility)
        realised_sharpe_ratio = np.mean(all_realised_sharpe_ratio)

        return weights, performance, realised_annual_return, realised_annual_volatility, realised_sharpe_ratio

    def shares_needed(self, aum, weights, test):
        """Calculate number of shares of each asset to be purchased given AUM and optimized portfolio allocation

        Args:
            aum (int): assets under management
            weights (pandas.Series): portfolio weights
            test (pandas.Series): test data

        Returns:
            [int]: number of shares of each asset to be purchased
        """
        investment = weights*aum
        number_of_shares = investment / test[0,:]
        return number_of_shares

    def plot_weights(self, weights):
        """Plot horizontal barchart of portfolio weights

        Args:
            weights (pandas.Series): portfolio weights
        """
        y = weights.index
        x = weights

        plt.barh(y, x)
        plt.ylabel("Assets")
        plt.xlabel("Weights")
        plt.title("Weights Horizontal Barplot")
        plt.show()
