import pandas as pd
import numpy as np
import sys
import pprint
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from util import lists_to_dict,sort_dict_by_value,extract_dict
import random
random.seed(314)

#RandomForestの分類予測をk-fold法で行い、その結果を返す関数
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

#指定した数の特徴量の重要度をソートしてdictで返す関数
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

    result = extract_dict(importances_dict_sorted,num)
    
    return result

#モデルによる予測結果の混同行列を表示する
def print_confusion_matrix(x,y,model,labels):
    '''
    引数：
    x=独立変数
    y＝目的変数
    model=トレーニング済みのモデル
    labels=混同行列のラベル（上から順番）
    ーーーーーーーー
    戻り値：
    混同行列
    （左上から順に
    [[TP,FN]
     [FP,TN]）
    '''
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    cm=confusion_matrix(y_test, y_pred,labels=labels)
    
    return cm

#予測結果と実際の結果を比較する
def compare_pred(x,y,model):
    '''
    引数：
    x=独立変数
    y＝目的変数
    model=トレーニング済みのモデル
    ーーーーーーーー
    戻り値：
    result=モデルに予測結果と実際の結果を比較したTrue/Falseのリスト
    '''
    y_test,y_pred=return_predict_true(x,y,model)
    result_list=y_pred==y_test

    return result_list

#予測結果と実際の結果を返す
def return_predict_true(x,y,model):
    '''
    引数：
    x=独立変数
    y＝目的変数
    model=トレーニング済みのモデル
    --------
    戻り値：
    y_test=実際の結果
    y_pred=モデルによる予測値
    '''
    y=np.reshape(y,-1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    
    return y_test,y_pred

def gridsearch(x,y,model,scoring):
    gridsearch = GridSearchCV(estimator = model,        # モデル
                          param_grid = param(),  # チューニングするハイパーパラメータ
                          scoring = scoring     # スコアリング
                             )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    gridsearch.fit(x_train, y_train)
    print('Best params: {}'.format(gridsearch.best_params_)) 
    print('Best Score: {}'.format(gridsearch.best_score_))