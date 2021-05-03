"""
YSC4228: Data Science in Quantitative Finance
Final Exam
Dragan Kantar

Run this script through a shell to sample a time series from a file.

Usage example: python get_dollar_bars.py --file e_mini_sp_series.csv --dollar_size 18000000
"""
import argparse
from dollar_bars import Sampler


def main():

    def enable_parsing(parser):
        """Adds the arguments to the parser

        Args:
            parser (Argument.Parser): parser

        Returns:
            [parser]: parser
        """
        parser.add_argument("--file", help="Time series data", required=True)
        parser.add_argument("--dollar_size", type=int, help="Dollar size", required=True)

        return parser

    parser = argparse.ArgumentParser()
    parser = enable_parsing(parser)
    args = parser.parse_args()

    print('\nArguments Parsed.\n')

    sampler = Sampler(args.file, args.dollar_size)

    print('\nLoading Data...\n')
    data = sampler.get_data()
    print('\nData loaded.\n')

    dollar_bars = sampler.get_dollar_bars(data)

    print('\nPlot 1: Dollar Bars Time Series\n')
    sampler.plot_one(dollar_bars)
    print('\nPlot 1 closed.')

    original_bars_weekly = sampler.resample_weekly(data)
    dollar_bars_weekly = sampler.resample_weekly(dollar_bars)

    print('\nPlot 2: Number Bars Produced by the Original and the Dollar Bars on a Weekly Basis\n')
    sampler.plot_two(original_bars_weekly, dollar_bars_weekly)
    print('\nPlot 2 closed.')

    print('Jarque-Bera Normality Test statistic for the original data: 'sampler.jarque_bera_test(data))
    print('Jarque-Bera Normality Test statistic for the dollar bars: 'sampler.jarque_bera_test(dollar_bars))


if __name__ == "__main__":
    try:
        main()
    except:
        print("Error encountered. Please try another command.")
