# optimize_portfolio.py --tickers GOOG,AAPL,FB --optimizer mvo -- au


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

objec = clas()


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
    objec.plot()