import pandas as pd
import sys
import pprint
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


def kfold_score_RandomForestClassifier(x,y,n_split):
    '''
    引数：
    x=x_train
    y=y_train
    n_split=k-fold法における分割の数
    ーーーー
    戻り値：
    scores_list=accuracyのlist
    predict_model=分類の予測モデル
    '''
    def kfold_score_RandomForestClassifier(x,y,n_split):
    cv = KFold(n_splits=n_split, random_state=1, shuffle=True)
    predicti_model = RandomForestClassifier()
    scores_list = cross_val_score(predicti_model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores_list,predicti_model