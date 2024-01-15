import pandas as pd
import numpy as np
import sys
import pprint
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from util import lists_to_dict,sort_dict_by_value
import random
random.seed(314)


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
    cv = KFold(n_splits=n_split, random_state=1, shuffle=True)
    predicti_model = RandomForestClassifier(random_state=1)
    scores_list = cross_val_score(predicti_model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
        
    return scores_list,predicti_model

def cul_importance(x,y,model,num):
    '''
    引数：
    x=モデルの独立変数
    y＝モデルの目的変数
    model=モデル
    num=表示したい特徴量の数（降順にsortを行う）
    ーーー
    戻り値
    result=特徴量名をkey,重要度をvalueとしたdict(valueを基準に降順sort済み)
    '''
    y=np.reshape(y,-1)
    labels = list(x.columns)
    model.fit(x,y)
    importances_list = model.feature_importances_
    importances_dict=lists_to_dict(labels,importances_list)
    importances_dict_sorted=sort_dict_by_value(importances_dict,reverse=True)

    result = {key:importances_dict_sorted[key] for key in list(importances_dict_sorted)[:num]} 
    
    return result