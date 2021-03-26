import os
os.chdir("C:/Users/draga/OneDrive - National University of Singapore/Electives/QuantFin/Assignment 2/script")
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")

train_x = train.iloc[:, 0]
train_y = train.iloc[:, 1]

test_x = test.iloc[:, 0]
test_y = test.iloc[:, 1]

f1 = make_scorer(f1_score)

xgb_parameters = {
                'xgboost__n_estimators': [150, 250, 350],
                'xgboost__max_depth': [15, 20, 25],
                'xgboost__learning_rate': [0.08, 0.09, 0.1],
                'xgboost__objective': ['binary:logistic'],
                'xgboost__use_label_encoder': [False],
                'xgboost__eval_metric': ['logloss']
                }

model = Pipeline(steps=[('tfidf', TfidfVectorizer()),
                        ('xgboost', xgb.XGBClassifier())])

gs_cv = GridSearchCV(estimator=model,
                     param_grid=xgb_parameters,
                     n_jobs=6,
                     refit=True,
                     cv=2,
                     scoring=f1)
gs_cv.fit(train_x, train_y)

gs_cv.best_params_
gs_cv.best_estimator_
gs_cv.best_score_

test_pred = gs_cv.predict(test_x)
