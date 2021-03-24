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


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from mlxtend.preprocessing import DenseTransformer

pipeline = Pipeline([
  ("vectorizer", TfidfVectorizer()),
  ("densifier", DenseTransformer()),
  ("classifier", XGBClassifier(random_state = 13))
])
pipeline.fit(X_train, y_train)



tfidf = TfidfVectorizer(max_features=self.max_feat)


train['vector']=vectorizer.fit_transform(train['item_name'])
train=train.drop('item_name',axis=1)
y=train.category_id
train=train.drop('category_id',axis=1)
X_train, X_test, y_train, y_test = train_test_split(train,y, test_size=0.10,stratify=y,random_state=42)
import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

########################################################################################
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import utils
import gensim.parsing.preprocessing as gsp
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("train.csv")
train_x = train[:, 0]
train_y = train[:, 1]
test = pd.read_csv("test.csv")

def clean_text(s):
    filters = [
           gsp.strip_tags,
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords,
           gsp.strip_short,
           gsp.stem_text
          ]
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s


class Text2TfIdfTransformer(BaseEstimator):

    def __init__(self):
        self._model = TfidfVectorizer(max_features=max_feat) # TODO: take care of max_feat
        pass

    def fit(self, train_x, df_y=None):
        train_x = train_x.apply(lambda x : clean_text(x))
        self._model.fit(train_x)
        return self

    def transform(self, train_x):
        return self._model.transform(train_x)


pl_xgb_tf_idf = Pipeline(steps=[('tfidf',Text2TfIdfTransformer()),
                                ('xgboost', xgb.XGBClassifier(objective=''))])

clf = GridSearchCV(pl_xgb_tf_idf, cv=10, n_jobs=-1)
cv_fit = clf.fit(pl_xgb_tf_idf, train_y)
pd.DataFrame(cv_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]


scores = cross_val_score(pl_xgb_tf_idf, train_x, train_y, cv=5)
print('Accuracy for Tf-Idf & XGBoost Classifier : ', scores.mean())

# TODO: how to parametrize XGB (incl objective, GridSearchCV)
# what is label encoder

# TODO: Testing
# TODO: outputing
# TODO: take care fo max_feat2