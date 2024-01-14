import pandas as pd
import pytest
from ..prediction import *
from sklearn.datasets import load_wine


def test_kfold_score_RandomForestClassifier():
    data=load_wine()
    x=data.data
    y=data.target
    score_list,predict_model=kfold_score_RandomForestClassifier(x,y,5)
    assert len(score_list)==5
    score_list,predict_model=kfold_score_RandomForestClassifier(x,y,10)
    assert len(score_list)==10
    
    score_list_1,predict_model=kfold_score_RandomForestClassifier(x,y,4)
    score_list_2,predict_model=kfold_score_RandomForestClassifier(x,y,4)
    #乱数が固定されているかテスト
    assert all(score_list_1==score_list_2)==True