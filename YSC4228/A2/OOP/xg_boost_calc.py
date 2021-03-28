import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import csv

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

class XG_Boost_Calculator():
    def __init__(self, num_folds, max_feat):
        self.num_folds = num_folds
        self.xgb_parameters = {
                'tfidf__max_features': [max_feat],
                'xgboost__n_estimators': [150],    #tried [50, 100, 140, 150, 160, 250, 350]
                'xgboost__max_depth': [15],        #tried [10, 13, 15, 17, 20, 25]
                'xgboost__learning_rate': [0.1],   #tried [0.09, 0.1, 0.12.]
                'xgboost__objective': ['binary:logistic'],
                'xgboost__use_label_encoder': [False],
                'xgboost__eval_metric': ['logloss']
                }

    def train(self, train):
        df_train = pd.read_csv(train)
        train_x = df_train.iloc[:, 0]
        train_y = df_train.iloc[:, 1]

        f1 = make_scorer(f1_score)

        model = Pipeline(steps=[('tfidf', TfidfVectorizer()),
                                ('xgboost', xgb.XGBClassifier())])

        gs_cv = GridSearchCV(estimator=model,
                            param_grid=self.xgb_parameters,
                            refit=True,
                            cv=self.num_folds,
                            scoring=f1)

        gs_cv.fit(train_x, train_y)
        return gs_cv

    def pred(self, test, gs_cv, pred):
        df_test = pd.read_csv(test)
        test_x = df_test.iloc[:, 0]
        test_y = df_test.iloc[:, 1]
        test_pred = gs_cv.predict(test_x)
        col_1 = test_x
        col_2 = test_pred
        try:
            with open(pred, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(list(col_1), list(col_2)))
        except:
            pass
