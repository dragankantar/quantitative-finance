#get_dollar_bars.py --file e_mini_sp_series.csv –-dollar_size 18000000
import argparse

def enable_parsing(parser):
    """
    Function that adds the arguments to our parser

    Returns: None
    """
    parser.add_argument("--file", help="Time series data", required=True)
    parser.add_argument("--dollar_size", help="Dollar size amount", required=True)

    return parser

parser = argparse.ArgumentParser()
parser = enable_parsing(parser)
args = parser.parse_args()





