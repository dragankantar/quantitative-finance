import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

class cleaner:
    def __init__(self, train, test, pred, max_feat, num_folds):
        self.train = pd.read_csv(train)
        self.test = pd.read_csv(test)
        self.train = xgb.DMatrix(train+'?format=csv&label_column=0')
        self.test = xgb.DMatrix(test+'?format=csv&label_column=0')
        self.pred = pred
        self.max_feat = max_feat
        self.num_folds = num_folds

        self.train = xgb.DMatrix(train+'?format=csv&label_column=0')
        self.test = xgb.DMatrix(test+'?format=csv&label_column=0')

    def xgboo():
        dtrain = xgb.DMatrix(data=self.train[:, 0], labels=self.train[:, 1])
        dtest = xgb.DMatrix(data=self.test[:, 0], labels=self.test[:, 1])

        num_round=10
        model = xgb.train(param, dtrain, num_round, evallist)

        pred = model.predict(dtest)

    def out(self):
        pred = pd.DataFrame(index=range(len(self.train)), columns=['Sentence', 'Predicted Bad Sentence'])
        pred.iloc[:, 0] = self.train[:, 0]
        pred.iloc[:, 1] = #predictions
        np.savetxt(self.pred, pred)

    def report_f1():
        return f1
