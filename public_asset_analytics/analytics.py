import argparse
from return_analytics import ReturnAnalytics

parser = argparse.ArgumentParser()
parser.add_argument("--ticker", type=str)
parser.add_argument("--period_begin", type=str)
parser.add_argument("--period_end", type=str)
args = parser.parse_args()

asset = ReturnAnalytics(args.ticker, args.period_begin, args.period_end)
