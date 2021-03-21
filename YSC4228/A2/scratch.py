# rain an XGBClassifier model with a TfidfVectorizer and then uses the resulting model to predict bad sentences

#clean_news.py --train train_data.csv --test test_data.csv --pred pred_data.csv  --max_feat 100 --num_folds 10

import argparse
from cleaner import cleaner

def parsing(parser):
    parser.add_argument("--train", help="Training data", required=True)
    parser.add_argument("--test", help="Testing data", required=True)
    parser.add_argument("--pred", help="Name of the output .csv file with predictions", required=True)
    parser.add_argument("--max_feat", type=int, help="Maximum number of features for TfidfVectorizer", required=True)
    parser.add_argument("--num_folds", type=int, help="Number of folds for k-fold cross-validation", required=True)
    return parser

parser = argparse.ArgumentParser()
parser = parsing(parser)
args = parser.parse_args()

text = cleaner(args.train, args.test, args.pred, args.max_feat, args.num_folds)

if args.pred: #do i even need the if statement if it is a required argument
    text.out()
#
#report F1 score
print("F1 Score is:", text.report_f1)
