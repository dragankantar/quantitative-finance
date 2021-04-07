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
