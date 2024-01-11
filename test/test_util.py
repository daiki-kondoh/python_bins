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

def test_extract_str_data():
    input_columns_list=['str0','str1','int0']
    input_data_list=[['data0','data0',0],
                   ['data1','data1',1],
                   ['data2','data2',0]]
    input_expect_columns_list=['str0','str1']
    input_expect_data_list=[['data0','data0'],
                   ['data1','data1'],
                   ['data2','data2']]
    sample_df=pd.DataFrame(data=input_data_list,columns=input_columns_list)
    expect_df=pd.DataFrame(data=input_expect_data_list,columns=input_expect_columns_list)
    diff_df=pd.concat([extract_str_data(sample_df),expect_df]).drop_duplicates(keep=False)
    #extract_str_data(sample_df)とexpect_dfが一致していれば、dfiif_dfの行数は0,列数はexpect_dfの列数(=2)になるはず
    assert diff_df.shape==(0, 2)
    
    input_data_list_2=[['data0','data0',0],
                   ['data1','data1','data1'],
                   ['data2','data2','data2']]
    sample_df_2=pd.DataFrame(data=input_data_list_2,columns=input_columns_list)
    diff_df=pd.concat([extract_str_data(sample_df_2),expect_df]).drop_duplicates(keep=False)
    assert diff_df.shape==(0, 2)

def test_extract_int_float_data():
    input_columns_list=['str0','float0','int0','float1','int1']
    input_data_list=[['data0',0.5,0,-1.0,-1],
                   ['data1',1.5,1,-2.0,-2],
                   ['data2',2.5,2,-3.0,-3]]
    input_expect_columns_list=['float0','int0','float1','int1']
    input_expect_data_list=[[0.5,0,-1.0,-1],
                   [1.5,1,-2.0,-2],
                   [2.5,2,-3.0,-3]]
    sample_df=pd.DataFrame(data=input_data_list,columns=input_columns_list)
    expect_df=pd.DataFrame(data=input_expect_data_list,columns=input_expect_columns_list)
    diff_df=pd.concat([extract_int_float_data(sample_df),expect_df]).drop_duplicates(keep=False)
    #extract_str_data(sample_df)とexpect_dfが一致していれば、dfiif_dfの行数は0,列数はexpect_dfの列数(=4)になるはず
    assert diff_df.shape==(0, 4)
    
    input_data_list_2=[[0,0.5,0,-1.0,-1],
                   ['data1',1.5,1,-2.0,-2],
                   ['data2',2.5,2,-3.0,-3]]
    sample_df_2=pd.DataFrame(data=input_data_list_2,columns=input_columns_list)
    diff_df=pd.concat([extract_int_float_data(sample_df_2),expect_df]).drop_duplicates(keep=False)
    assert diff_df.shape==(0, 4)

    def test_value_to_dummy():
    dic={'?':'NaN','NaN':'--','a':1}
    assert value_to_dummy('?',dic)=='NaN'
    assert value_to_dummy('NaN',dic)=='--'
    assert value_to_dummy('a',dic)==1
    assert value_to_dummy(0,dic)==0


def test_df_value_to_dummy():
    input_columns_list=['test1-0']
    input_data_list=[[1],
                     [2],
                     ['?'],
                     [4]]
    input_expect_columns_list=['test1-0']
    input_expect_data_list=[[1],
                     [2],
                     ['NA'],
                     [4]]
    sample_df=pd.DataFrame(data=input_data_list,columns=input_columns_list)
    expect_df=pd.DataFrame(data=input_expect_data_list,columns=input_expect_columns_list)
    dic={'?':'NA'}
    diff_df=pd.concat([df_value_to_dummy(sample_df,dic),expect_df]).drop_duplicates(keep=False) 
    assert diff_df.shape==(0, 1)
    
    input_columns_list_2=['test2-0']
    input_data_list_2=[[1],
                     [2],
                     ['?'],
                     ['?']]
    input_expect_columns_list_2=['test2-0']
    input_expect_data_list_2=[[1],
                     [2],
                     ['NA'],
                     ['NA']]
    sample_df=pd.DataFrame(data=input_data_list_2,columns=input_columns_list_2)
    expect_df=pd.DataFrame(data=input_expect_data_list_2,columns=input_expect_columns_list_2)
    dic={'?':'NA'}
    diff_df=pd.concat([df_value_to_dummy(sample_df,dic),expect_df]).drop_duplicates(keep=False) 
    assert diff_df.shape==(0, 1)
    
    input_columns_list_3=['test3-0']
    input_data_list_3=[[1],
                     [2],
                     ['?'],
                     ['---']]
    input_expect_columns_list_3=['test3-0']
    input_expect_data_list_3=[[1],
                     [2],
                     ['NA'],
                     ['NA']]
    sample_df=pd.DataFrame(data=input_data_list_3,columns=input_columns_list_3)
    expect_df=pd.DataFrame(data=input_expect_data_list_3,columns=input_expect_columns_list_3)
    dic={'?':'NA','---':'NA'}
    diff_df=pd.concat([df_value_to_dummy(sample_df,dic),expect_df]).drop_duplicates(keep=False) 
    assert diff_df.shape==(0, 1)

    input_columns_list_4=['test4-0']
    input_data_list_4=[[1],
                     [2],
                     ['?'],
                     ['---']]
    input_expect_columns_list_4=['test4-0']
    input_expect_data_list_4=[[1],
                     [2],
                     ['NA'],
                     ['']]
    sample_df=pd.DataFrame(data=input_data_list_4,columns=input_columns_list_4)
    expect_df=pd.DataFrame(data=input_expect_data_list_4,columns=input_expect_columns_list_4)
    dic={'?':'NA','---':''}
    diff_df=pd.concat([df_value_to_dummy(sample_df,dic),expect_df]).drop_duplicates(keep=False) 
    assert diff_df.shape==(0, 1)