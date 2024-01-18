import pandas as pd
import pytest
from ..prediction import *
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris


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


def test_cul_importance():
    data=load_wine()
    x=pd.DataFrame(data.data)
    y=pd.DataFrame(data.target)
    score_list_1,model=kfold_score_RandomForestClassifier(x,y,2)
    assert  len(cul_importance(x,y,model,5))==5
    assert  len(cul_importance(x,y,model,5).keys())==5
    assert  max(list(cul_importance(x,y,model,5).values()))==list(cul_importance(x,y,model,5).values())[0]
    assert  min(list(cul_importance(x,y,model,5).values()))==list(cul_importance(x,y,model,5).values())[4]


def test_print_confusion_matrix():
    data=load_iris()
    x=data.data
    y=data.target
    expect=[[13,  0,  0],
            [ 0, 15,  1],
            [ 0,  0,  9]]
    labels=[0,1,2]
    score_list,model=kfold_score_RandomForestClassifier(x,y,5)
    assert np.all(print_confusion_matrix(x,y,model,labels=labels)==expect,keepdims=True)==True
    
    data2=load_wine()
    x2=data2.data
    y2=data2.target
    expect2=[[18,  0,  0],
             [ 1, 16,  0],
             [ 0,  0, 10]]
    labels=[0,1,2]
    score_list,model2=kfold_score_RandomForestClassifier(x2,y2,5)
    assert np.all(print_confusion_matrix(x2,y2,model2,labels=labels)==expect2,keepdims=True)==True