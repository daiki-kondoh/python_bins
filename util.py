# util.py

import pandas as pd
import numpy as np
from IPython.core.display import display

#各列の欠損値を表示する関数
def display_null_number(df):
  null_df=pd.DataFrame(df.isnull().sum()).copy()
  null_df['null数']=null_df[0]
  
  return null_df[['null数']]


#各列に格納されている値のTypeの種類と数をdict型でdfに格納して形で返す関数
def type_values(df):
  df_types=df.copy()
  cols=df_types.columns
  df_type_dict=pd.DataFrame(np.arange(df_types.shape[1]).reshape(df_types.shape[1], 1),index=cols)
  df_type_dict['type:数']=df_type_dict[0]
  for i in range(0,len(cols)):
    types_list=list(range(df_types.shape[0]))
    for j in range(0,df_types.shape[0]):
      value_type=str(type(df_types.at[df_types.index[j],cols[i]]))
      types_list[j]=value_type
    df_types[cols[i]]=types_list
    type_dict=df_types[cols[i]].value_counts().to_dict()
    df_type_dict['type:数'][i]=type_dict
  df_type_dict=df_type_dict[['type:数']]

  return df_type_dict


#各列に格納されている値のTypeの種類と数を一つずつ表示する関数
def display_type_values_by_cols(df):
  cols=df.columns
  for i in range(df.shape[1]):
    types=pd.DataFrame(df[cols[i]].apply(lambda x:type(x)).value_counts()).T
    display(types)


#任意の列の値を結合して、新しいIdを作成する関数
def create_id(df,cols):
#colsは結合したい列名のリスト
  df_create_id=df[cols].copy()
  df_create_id['new_id']=df_create_id[cols[0]].astype('str')
  for i in range(1,len(cols)):
    df_create_id[cols[i]]=df_create_id[cols[i]].astype('str')
    df_create_id['new_id']=df_create_id['new_id'].str.cat(df_create_id[cols[i]])
  
  df_create_id[['new_id']]
  df_create_id=pd.concat([df_create_id[['new_id']],df],axis=1)
  df_create_id

  return df_create_id

#クロス集計した時の各行の値を一つのパターンとしてdict型で返す関数
def pattern_crosstab_to_dict(df,crosstab_row,crosstab_col):
  df_pattern_crosstab=pd.crosstab(crosstab_row, crosstab_col)
  pattern_dict=df_pattern_crosstab.to_dict(orient='index')
  
  return pattern_dict


#クロス集計した各パターンの値をdict型のstrとして保持するDataFrameを返す関数
def pattern_crosstab_to_df(df,crosstab_row,crosstab_col):
  df_pattern_crosstab=pd.crosstab(crosstab_row, crosstab_col)
  pattern_dict=df_pattern_crosstab.to_dict(orient='records')
  dict_num=len(pattern_dict)
  pattern_list=list(range(dict_num))
  for i in range(dict_num):
    pattern_list[i]=pattern_dict[i]
    df_pattern_crosstab[f'{crosstab_col.name}_pattern']=pattern_list

  display(df_pattern_crosstab[[f'{crosstab_col.name}_pattern']])

  return df_pattern_crosstab[[f'{crosstab_col.name}_pattern']]


#dataFrameの各列の最大の文字列/桁数をdataframe型で返す関数
def df_max_digits(df):
  cols=df.columns
  digits_list=list(range(len(cols)))
  for i in range(len(cols)):
    digits_list[i]=series_max_digits(df[cols[i]])
  df_digits=pd.DataFrame(digits_list,index=cols)

  return df_digits


#Series型のデータの中で最大の文字列/桁数を返す関数
def series_max_digits(data_series):
    """
    pandasのSeriesが与えられたとき、Seriesが数値型の場合は数値の最大桁数を返し、文字列型の場合は最大文字数を返す。

    引数:
        series (pd.Series): pandasのSeries

    戻り値:
        int: Seriesの要素の中の最大の桁数もしくは文字数
    """
    # seriesが空の場合は0を返す
    if len(data_series) == 0:
        return 0

    # seriesの型に応じて処理を分ける
    # seriesが数値型の場合は数値の桁数を返す
    if data_series.dtype == 'int64':
        return int(len(data_series.abs().max().astype(str)))
    # seriesが浮動小数点型の場合は小数点以下の桁数を返す
    elif data_series.dtype == 'float64':
        return int(data_series.abs().astype(str).str.len().max()-1)
    # seriesが文字列型の場合は文字数を返す
    elif data_series.dtype == 'object':
       return data_series.str.len().max()