"""
YSC4228: Data Science in Quantitative Finance
Midterm Assignment
Dragan Kantar
"""

import argparse
import numpy as np
from calculator import returns_calc

def parsing(parser):
    parser.add_argument("--ticker")
    parser.add_argument("--b", type=str)
    parser.add_argument("--e", type=str)
    parser.add_argument("--initial_aum", type=np.float64)
    parser.add_argument("--plot", action='store_true')
    return parser

parser = argparse.ArgumentParser()
parser = parsing(parser)
args = parser.parse_args()

stock = returns_calc(args.ticker, args.b, args.e, args.initial_aum)

print("Company Name:", stock.get_name())

print("Begin Date:", stock.begin_real)
print("End Date:", stock.end_real)
print("Number of Calendar Days:", stock.calendar_days(), "days.")

print("Total Stock Return (adj for dividends):", stock.total_stock_return().round(2))
print("Total Return (of the AUM invested):", 100*stock.total_return_aum().round(2), "percent.")
print("Annualized rate of return (of the AUM invested):", 100*stock.annualized_ror_calc().round(4), "percent.")

print("Initial AUM:", args.initial_aum.round(2))
print("Final AUM:", stock.final_aum_calc().round(2))
print("Average AUM:", stock.avg_aum_calc().round(2))
print("Maximum AUM:", stock.max_aum_calc().round(2))

print("PnL (of the AUM invested):", stock.pnl_calc().round(2))
print("Avg Daily Return of the Portfolio:", 100*stock.avg_daily_ret().round(6), "percent.")
print("Daily Standard Deviation of the Portfolio Return:", stock.sd_portfolio_d().round(4))
print("Daily Sharpe Ratio of the portfolio (assuming risk-free rate of 0.01pct):", stock.sharpe_calc().round(2))

if args.plot:
    stock.plot()
