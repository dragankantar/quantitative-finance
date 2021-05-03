import argparse
from dollar_bars import Sampler

def enable_parsing(parser):
    """
    Function that adds the arguments to the parser

    Returns: parser
    """
    parser.add_argument("--file", help="Time series data", required=True)
    parser.add_argument("--dollar_size", type=int, help="Dollar size", required=True)

    return parser

parser = argparse.ArgumentParser()
parser = enable_parsing(parser)
args = parser.parse_args()

sampler = Sampler(args.file, args.dollar_size)

data = sampler.get_data()

# plot 1
dollar_bars = sampler.get_dollar_bars(data)
sampler.plot_one(dollar_bars)

#sampler.plot_one(data)

# plot 2
original_bars_weekly = sampler.resample_weekly(data)
dollar_bars_weekly = sampler.resample_weekly(dollar_bars)
sampler.plot_two(original_bars_weekly, dollar_bars_weekly)

#sampler.plot_two(dollar_bars_weekly)

# JB test
print(sampler.jarque_bera_test(data))
print(sampler.jarque_bera_test(dollar_bars))
