# optimize_portfolio.py --tickers GOOG,AAPL,FB --optimizer mvo -- au

"""
Hints:

Please get all details on how to use the API hereLinks to an external site.. Explanations are really good.

Tickers should be a comma-separated strings with no spaces (e.g., ‘GOOG,AAPL,FB’). You would then split it on the code and get the tickers.

The program will get tickers prices as before – I recommend you do this once, save it into a file, and do all the testing with that file so that you don’t get blocked from Yahoo or whatever API you are using.

‘optimizer’ is just the optimizer to be used. There are three options:
msr: Maximum Sharpe Ratio
mvo: Mean-Variance Optimization
hrp: Hierarchical Risk Parity
"""

"""
Hi All,

Please see below a Q&A regarding the assignment. It is extracted from some emails I've gotten from different teams. Hopes this helps to clarify things out. I'll be speaking more about it on Thursday as this is a very important and practical assignment.

If I understand correctly, we use Pyportfolioopt to get the best asset allocation for today (and the expected stats). Yes, that is correct.

What data should we use for this: all the available data or just the previous n months (e.g. 12)? Since it is for today, we can use all available data as of today.

After doing so, we need to perform the backtesting. My understanding is that the 'strategy' we are testing is the pyportfolioopt optimizer (is that correct?).Yes, that is correct.

We are testing different allocation strategies. By "starting at the end", do you mean something like: given that we buy at the end of month 11, what would our expected returns be? Subsequently, given that we buy at the end of month 10, what returns do we get, and so on? Exactly. To be more specific, when doing backtesting, we only care about what we got (i.e., the realised return and other stats) 

Again, are we using all available data or only data from given months to do this? Only the available data that was available at the time. For example, if you are at the end of May of last year, you can only use data for May and before. You have to pretend that you don’t know the future. That's the whole idea of backtesting.

When you say to use the same parameters, what parameters are you referring to? The same portfolio optimization technique, history length (e.g. last 12 months) and any other parameter (e.g. gamma)

The returns and covariance matrix that we feed the EfficientFrontier will change for each backtest (as the data changes). Yes, that’s correct

The backtesting will give a different set of outputs for every month. Which one do you expect us to print? The aggregated realized results.

We are a bit confused about the difference between the expected statistics and the backtest statistics? Won't both be based on the historical data retrieved for the requested tickers? Based on the example given, it also seems that they provide the same values for variables like annual return, annual volatility, and annual Sharpe ratio. Say you pretend to be at the end of March last year. You can use all the historical information before that and construct your optimal portfolio. ef.portfolio_performance(verbose=True) will give you the expected statistics for the next year. Thes are just expected values (not real) based on data on or before March last year. Now, since we already have the data between March last year and March this year, we can compute the actual (backtest or realised) statistics. It will give us an idea on how different the predictions were from reality. This is the whole idea of backtesting.

We are also confused regarding the difference between mean-variance optimization and maximum Sharpe ratio. According to PyPortfolioOpt documentation, it seems that mean-variance optimization is under EfficientFrontier which allows to take in arguments like min_volatility() (which is what we have used so far for MVO) and max_sharpe (which we have used for Maximum Sharpe Ratio). Is this the correct interpretation of the instructions? Yes, this is correct.

Thanks to all teams asking these interesting questions.

Kelvin

"""

import argparse
from pypfopt.objective_functions import portfolio_return

from sklearn.utils import shuffle

def parsing(parser):
    parser.add_argument("--tickers", help="Assets tickers", required=True)
    parser.add_argument("--optimizer", type=str, help="Portfolio optimizer", required=True)
    parser.add_argument("--aum", type=float, help="Assets under Management", required=True)
    parser.add_argument("--backtest_months", type=int, help="Number of months for which the program performs a backtest", required=True)
    parser.add_argument("--plot_weights", type=int, help="Whether to plot portfolio weights as a horizontal bar chart",
                        required=True, action='store_true')
    return parser

parser = argparse.ArgumentParser()
parser = parsing(parser)
args = parser.parse_args()

portfolio = OptimizerBacktester()


print("Backtest Stats:")
print("Start Date:")
print("End Date:")
print("Annual Return:")
print("Annual Volatility:")
print("Annual Sharpe Ratio:")

print("Expected Stats:")
print("Annual Return:")
print("Annual Volatility:")
print("Annual Sharpe Ratio:")

print("Shares Needed:")
for i in range (tick):
    print(tickers[i], ":", no_shares[i])

if args.plot_weights:
    portfolio.plot_weights()


#######################################################
#Alaukik's start

import datetime 
tod = datetime.datetime.now()
d = datetime.timedelta(days = 1)
a = tod - d
backtest_type = "wf"

tickers = ['BA', 'AMD', 'AAPL']
start = dt.datetime(2021, 3, 1)
end = dt.datetime(2021, 3, 30)
ohlc = yf.download(tickers, start=start, end=end)
prices = ohlc["Adj Close"].dropna(how="all")
df = prices


from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

mu = mean_historical_return(df)
S = CovarianceShrinkage(df).ledoit_wolf()
from pypfopt.efficient_frontier import EfficientFrontier

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
# ef.save_weights_to_file("weights.txt")  # saves to file
print(cleaned_weights)

ef.portfolio_performance(verbose=True);

#####################################################################################
# an example from pyportfolioopt

# if you already have expected returns and risk model
from pypfopt.efficient_frontier import EfficientFrontier

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()

# if you do not

import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Read in price data
df = pd.read_csv("tests/resources/stock_prices.csv", parse_dates=True, index_col="date")

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)

#########################################

from sklearn.model_selection import train_test_split, ShuffleSplit
import numpy as np
X, y = np.arange(10).reshape((5, 2)), range(12)
list(y)
train_test_split(y, test_size = 1/12, train_size = 11/12, shuffle=False)
next(ShuffleSplit().split(y))

len=120
btm = 12
list(range(btm))

2*len/btm + len/btm-1




df = 0:119
test1 = 0:9
test2 = 10:19
test3 = 20:29
test4 = 30:39
test5 = 40:49
test6 = 50:59
test7 = 60:69
test8 = 70:79
test9 = 80:89
test10 = 90:99
test11 = 100:109
test12 = 110:119

from _typeshed import StrPath
import numpy as np
import matplotlib.pyplot as plt
import statistics
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
import pandas_datareader as web
import yfinance as yf
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt




end_date = datetime.date.today()
end_date = datetime.date(2021, 3, 31)
start_date = end_date - relativedelta(months=1)
back_test_date = end_date - relativedelta(months=12)

df = yf.download("AAPL,GOOG,FB", start=back_test_date, end=start_date)
df = df["Adj Close"].dropna(how="all")

test_df = yf.download("AAPL,GOOG,FB", start=start_date, end=end_date)
test_df = test_df["Adj Close"].dropna(how="all")

returns = df.pct_change().dropna()
hrp = HRPOpt(returns)
weights = hrp.optimize()
performance = hrp.portfolio_performance(verbose=True)

    optimizer = "mvo"
    mu = mean_historical_return(df)
    S = CovarianceShrinkage(df).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe() if optimizer == "msr" else ef.min_volatility()
    cleaned_weights = ef.clean_weights() # maybe remove this bc simplicity
    performance = ef.portfolio_performance()


weights = pd.Series(weights)
total_asset_returns = (test_df.iloc[-1]-test_df.iloc[0])/test_df.iloc[0]
test_months = 1

realized_annual_returns = (test_df.iloc[-1]/test_df.iloc[0])**(12/test_months)-1



annual_returns = (test_df.iloc[-1]/test_df.iloc[0])**(12/test_months)-1
weights*annual_returns
portfolio_returns = sum(weights*total_asset_returns)


sum(weights*((test.iloc[-1]/test.iloc[0])**(12/test_months)-1))
sum(np.std(test_df)*weights)



aum = 10000
investment = weights*aum
number_of_shares = investment / test_df.iloc[0,:]
number_of_shares


train_returns = train.pct_change().dropna()
hrp = HRPOpt(train_returns)
weights = hrp.optimize()
weights = pd.Series(weights)

performance = hrp.portfolio_performance(verbose=True)

back_test_months = 12
all_weights = np.zeros((back_test_months, np.shape(df)[1]))
all_weights[0] = weights

weights.index[0]

weights[0]

number_of_shares


df_index = list(range(len(df)))
df.reindex(df_index)
df

pd.to_numeric(df.index, downcast='float')
# although normally I would prefer an integer, and to coerce errors to NaN
pd.to_numeric(df.index, errors = 'coerce',downcast='integer')

df.index


df["index"] = df_index

type(df_index[0])



#data_index = list(range(len(data)))
#data["index"] = data_index
#data.set_index("index")



end_date = datetime.date.today()
start_date = end_date - relativedelta(months=1)
back_test_date = end_date - relativedelta(months=12)

back_test_months
i=0
embargo = int(np.round_(0.01*len(data), decimals=0))
data = yf.download("AAPL,GOOG,FB", start=back_test_date, end=end_date)
data = data["Adj Close"].dropna(how="all")

test_start = i*len(data)/back_test_months
test_start
test_end = test_start+len(data)/back_test_months-1
test_end
test = data.iloc[int(test_start):int(test_end), :]
test
train = data.iloc[np.r_[0:int(test_start), int(test_end)+int(embargo):len(data)], :]
train


data.iloc[np.r_[0:5, 150:155], :]


data.iloc[200:-1, :]

all_weights = np.zeros((back_test_months, np.shape(data)[1]))





###
annual_risk_free_rate = 0.02
optimizer = "mvo"
embargo = np.round_(0.01*len(data), decimals=0)
all_weights = np.zeros((back_test_months, np.shape(data)[1]))
all_realised_annual_return = np.zeros(back_test_months)
all_realised_annual_volatility = np.zeros(back_test_months)
all_realised_sharpe_ratio = np.zeros(back_test_months)

for i in range(back_test_months):

    test_start = i*len(data)/back_test_months
    test_end = test_start+len(data)/back_test_months-1
    test = data.iloc[int(test_start):int(test_end), :]
    train = data.iloc[np.r_[0:int(test_start), int(test_end)+int(embargo):len(data)], :]
    #print("test", test)
    #print("train", train)
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

    all_realised_annual_return[i] = sum(all_weights[i]*((test.iloc[(len(test)-1)]/test.iloc[0])**(12/test_months)-1))
    all_realised_annual_volatility[i] = sum(all_weights[i]*np.std(test.pct_change().dropna())*np.sqrt(12))
    all_realised_sharpe_ratio = (all_realised_annual_return[i]-annual_risk_free_rate)/all_realised_annual_volatility[i]

weights = np.mean(all_weights)
realised_annual_return = np.mean(all_realised_annual_return)
realised_annual_volatility = np.mean(all_realised_annual_volatility)
realised_sharpe_ratio = np.mean(all_realised_sharpe_ratio)
