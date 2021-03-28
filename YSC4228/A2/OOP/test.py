from numpy.testing import assert_approx_equal
from xg_boost_calc import XG_Boost_Calculator

class Test():

    def __init__(self, train, test, pred, max_feat, num_folds):
        self.test = test
        self.pred = pred
        self.calculator = XG_Boost_Calculator(num_folds, max_feat)
        self.gs_cv = self.calculator.train(train)
        self.f1_score = self.gs_cv.best_score_


    def test_f1_score(self):
        assert_approx_equal(self.f1_score, 0.761683981198004, significant=2)

    def test_output_file(self):
        self.calculator.pred(self.test, self.gs_cv, self.pred)
        try:
            f = open(self.pred)
        except IOError:
            print("No result written")
            assert(false)
        finally:
            f.close()


if __name__ == "__main__":
    TRAIN = "train_data.csv"
    TEST = "test_data.csv"
    PRED = "pred_data.csv"
    max_feat = 5
    num_folds = 5
    print("Testing...")
    tester = Test(TRAIN, TEST, PRED, max_feat, num_folds)
    tester.test_f1_score()
    tester.test_output_file()
    print("All tests passed!")
