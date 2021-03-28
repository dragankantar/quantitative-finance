"""
YSC4228: Data Science in Quantitative Finance
Group Assignment 2 - NLP in Finance
"""
import argparse

from xg_boost_calc import XG_Boost_Calculator

def main():

    def enable_parsing(parser):
        parser.add_argument("--train", help="Training data", required=True)
        parser.add_argument("--test", help="Testing data", required=True)
        parser.add_argument("--pred", help="Name of the output .csv file with predictions", required=True)
        parser.add_argument("--max_feat", type=int, help="Maximum number of features for TfidfVectorizer", required=True)
        parser.add_argument("--num_folds", type=int, help="Number of folds for k-fold cross-validation", required=True)
        return parser

    parser = argparse.ArgumentParser()
    parser = enable_parsing(parser)
    args = parser.parse_args()

    TRAIN = args.train
    TEST = args.test
    PRED = args.pred
    max_feat = int(args.max_feat)
    num_folds = int(args.num_folds)

    print("Parsed arguments.. \n")

    calculator = XG_Boost_Calculator(num_folds, max_feat)
    print("Initialized model..\n")


    gs_cv = calculator.train(TRAIN)
    print("Calculated best parameters..\n")

    calculator.pred(TEST, gs_cv, PRED)
    print("Wrote to file " + PRED + "\n")

    #report F1 score
    print("F1 Score is:", gs_cv.best_score_)

if __name__ == "__main__":
    try:
        main()
    except:
        print("Please provide new command. Did not accept current command.")
