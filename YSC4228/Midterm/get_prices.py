"""
YSC4228: Data Science in Quantitative Finance
Midterm Assignment
Dragan Kantar

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yfinance as yf
from datetime import date

parser = argparse.ArgumentParser()
parser.add_argument("--ticker", nargs=1, type=str)
parser.add_argument("--b", nargs=1, type=str)
parser.add_argument("--e", nargs=1 , type=str)
parser.add_argument("--initial_aum", nargs=1, type=int)
parser.add_argument("--plot", action='store_true')
args = parser.parse_args()

ticker = args.ticker
start = args.b
end = args.e
initial_aum = args.initial_aum

data = yf.download(ticker, begin, end)
data = data['Adj Close']

trading_days = data.shape[0]
begin_date = date(int(begin[0:4]), int(begin[4:6]), int(begin[6:]))
end_date = date(int(end[0:4]), int(end[4:6]), int(end[6:]))
calendar_days = end_date-begin_date
num_years = calendar_days.days/356

total_ret = (data[-1]-data[0])/data[0]
net_ret = data[-1]-data[0]
total_ret_pct = total_ret*100
total_ret_aum = args.initial_aum*total_ret

annualized_ror = ((initial_aum + net_ret)/initial_aum)**(1/num_years)-1
annualized_ror_pct = annualized_ror*100

final_aum = initial_aum + initial_aum*total_ret

daily_ret = np.zeros(len(data))
daily_cumul_ret = np.zeros(len(data))
for i in range(len(data)-1):
    daily_ret[i] = (data[i+1]-data[i])/data[i]
    daily_cumul_ret[i] = (data[i]-data[0])/data[0]
np.mean(daily_ret)
np.mean(daily_cumul_ret) #not needed?

current_aum = np.zeros(len(data))
current_aum[0] = initial_aum
for i in range(len(data)-1):
    current_aum[i+1] = current_aum[i] + daily_ret[i] * current_aum[i]
avg_aum = np.mean(current_aum)
max_aum = max(current_aum)

pnl = final_aum-initial_aum

rets_d = data.pct_change()
mean_rets_d = rets_d.mean()
mean_rets_d_pct = mean_rets_d*100

sd_rets_d = np.std(rets_d)

sharpe = (mean_rets_d_pct - 0.01)/sd_rets_d


print("Begin Date:", args.b)
print("End Date:", args.e)
print("Number of Days:", calendar_days.days) # trading days or calendar days

print("Total Stock Return (adj for dividends):", total_ret.pct.round(2), "percent") # total as percentage
print("Total Return (of the AUM invested):", total_ret_aum.round(2))
print("Annualized rate of return (of the AUM invested):", annualized_ror_pct.round(2), "percent")

print("Initial AUM:", args.initial_aum)
print("Final AUM:", final_aum)
print("Average AUM:", avg_aum)
print("Maximum AUM:", max_aum)

print("PnL (of the AUM invested):", pnl) # is this a the end?
print("Avg Daily Return of the Portfolio:", mean_rets_d_pct, "percent")
print("Daily Standard Deviation of the Portfolio Return:", sd_rets_d)
print("Daily Sharpe Ratio of the portfolio (assuming risk-free rate of 0.01pct):", sharpe)

if args.plot:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(data.index, current_aum)
    ax.set_xlabel('Time')
    ax.set_ylabel('AUM')
    plt.show()
