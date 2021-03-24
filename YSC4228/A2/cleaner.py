import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

class cleaner:
    def __init__(self, train, test, pred, max_feat, num_folds):
        self.train = xgb.DMatrix(data=train[:, 0], labels=train[:, 1])
        self.test = xgb.DMatrix(data=test[:, 0], labels=test[:, 1])
        self.pred = pred
        self.max_feat = max_feat
        self.num_folds = num_folds

    def xgboo():
        num_round=10
        model = xgb.train(param, self.train, num_round, evallist)

        pred = model.predict(self.test)

    def out(self):
        pred = pd.DataFrame(index=range(len(self.train)), columns=['Sentence', 'Predicted Bad Sentence'])
        pred.iloc[:, 0] = self.train[:, 0]
        pred.iloc[:, 1] = #predictions
        np.savetxt(self.pred, pred)

    def report_f1():
        return f1
