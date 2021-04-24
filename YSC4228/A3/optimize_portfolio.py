"""
YSC4228: Data Science in Quantitative Finance
Assignment 3
Alaukik Pant, Dragan Kantar, Martynas Galnaitis

Run this script to do the backtesting for portfolio allocation

usage: run the script through a shell, specifying arguments: tickers, optimizer, aum, backtest_months, test_months, backtest_type, plot_weights (optional)

available optimizer options: msr (Maximum Sharpe Ratio), mvo (Mean-Variance Optimization), hrp (Hierarchical Risk Parity)
available backtest types: wf (Walk-Forward), cv (Cross-Validation)

usage example: python optimize_portfolio.py --tickers GOOG,AAPL,FB --optimizer mvo --aum 10000 --backtest_months 12 --test_months 1 --backtest_type wf --plot_weights

"""

import argparse
from backtester import BackTester
import numpy as np

def main():

    def enable_parsing(parser):
        """
        Function that adds the arguments to our parser

        Returns: None
        """
        parser.add_argument("--optimizer", choices=['msr', 'mvo', 'hrp'], help="Optimizers: msr, mvo, hrp", required=True)
        parser.add_argument("--backtest_type", choices=['wf', 'cv'], help="Backtest type options: wf, sv", required=True)
        parser.add_argument("--tickers", help="Tickers to be used seperated by comma", required=True)
        parser.add_argument("--aum", type=int, help="Assets Under Management", required=True)
        parser.add_argument("--backtest_months", type=int, help="Backtest months to be used", required=True)
        parser.add_argument("--test_months", type=int, help="Test months to be used", required=True)
        parser.add_argument('--plot_weights', help="To plot",
                                action='store_true')
        return parser

    parser = argparse.ArgumentParser()
    parser = enable_parsing(parser)
    args = parser.parse_args()

    aum = args.aum
    optimizer = args.optimizer
    backtest_type = args.backtest_type
    tickers = args.tickers
    tickers = tickers.split(",")
    backtest_months = args.backtest_months
    test_months = args.test_months
    TO_PLOT = args.plot_weights

    if ((backtest_months % test_months) != 0):
        print("Backtest month should be a multiple of test months.")
        return


    backtest = BackTester(tickers)

    start_date, end_date, back_test_date = backtest.get_dates(test_months, backtest_months)
    train, test, data =  backtest.get_data(back_test_date, start_date, end_date)
    weights, performance, realised_annual_return, realised_annual_volatility, realised_sharpe_ratio = backtest.wf(train, optimizer, test, test_months) if backtest_type == "wf" else backtest.cv(backtest_months, data, optimizer, test_months, annual_risk_free_rate=0.02)
    expected_annual_return, annual_volatility, sharpe_ratio = performance
    number_of_shares = backtest.shares_needed(aum, weights, test)

    print("\nBacktest Statistics\n")
    print("Start date is " + start_date.strftime('%Y-%m-%d'))
    print("End date is " + end_date.strftime('%Y-%m-%d'))
    print("Realised Annual Return: " + str(np.round(realised_annual_return, decimals=4)*100) + "%")
    print("Realised Annual Volatility: " + str(np.round(realised_annual_volatility, decimals=4)*100) + "%")
    print("Realised Sharpe Ratio: " + str(np.round(realised_sharpe_ratio, decimals=4)))

    print("\nExpected Statistics\n")
    print("Annual Return: " + str(np.round(expected_annual_return, decimals=4)*100) + "%")
    print("Annual Volatility: " + str(np.round(annual_volatility, decimals=4)*100) + "%")
    print("Annual Sharpe Ratio: " + str(np.round(sharpe_ratio, decimals=4)))

    print("\nShares Needed\n")
    print(str(number_of_shares))

    if TO_PLOT:
        backtest.plot_weights(weights)

    return


if __name__ == "__main__":
    try:
        main()
    except:
        print("Please provide new command. Did not accept current command.")
