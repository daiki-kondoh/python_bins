import pandas as pd
import pytest
from ..util import *

def test_lists_to_dict():
    input_key_list=['test_key1','test_key2']
    input_value_list=[1,2]
    
    input_key_list_2=['test_key1','test_key2','test_key3']
    input_value_list_2=[1,2,3]
    
    assert lists_to_dict(input_key_list,input_value_list)=={'test_key1':1,'test_key2':2}
    assert lists_to_dict(input_key_list,input_value_list)=={input_key_list[0]:input_value_list[0],input_key_list[1]:input_value_list[1]}
    assert lists_to_dict(input_key_list_2,input_value_list_2)=={input_key_list_2[0]:input_value_list_2[0],input_key_list_2[1]:input_value_list_2[1],input_key_list_2[2]:input_value_list_2[2]}
    

def test_type_value():
    input_column_list=['test']
    input_data_list=['data',2,'']
    sample_df=pd.DataFrame(data=input_data_list,columns=input_column_list)
    
    assert type_value(sample_df)=={'test':{str:2,int:1}}
    
    input_column_list_2=['test0','test1','test2']
    input_data_list_2=[['data',2.2,'a'],
                   [2,2.3,'b'],
                   ['',1.5,'c']]
    sample_df_2=pd.DataFrame(data=input_data_list_2,columns=input_column_list_2)
    
    assert type_value(sample_df_2)=={'test0':{str: 2, int: 1}, 'test1':{float: 3}, 'test2':{str: 3}}
