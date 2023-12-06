# util.py

import pandas as pd
import numpy as np
from IPython.core.display import display

#各列の欠損値を表示する関数
def display_null_number(df):
  null_df=pd.DataFrame(df.isnull().sum().sort_values(ascending=False)).copy()
  null_df['null数']=null_df[0]
  display(null_df[['null数']][null_df['null数']>0])


#各列に格納されている値のTypeの種類と数をdfの形で表示する関数
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


#クロス集計した時の各行の値を一つのパターンとしてdict型で保存する関数
def pattern_crosstab(df,crosstab_row,crosstab_col):
  df_pattern_crosstab=pd.crosstab(crosstab_row, crosstab_col)
  pattern_dict=df_pattern_crosstab.to_dict(orient='records')
  dict_num=len(pattern_dict)
  pattern_list=list(range(dict_num))
  for i in range(dict_num):
    pattern_list[i]=pattern_dict[i]
    df_pattern_crosstab[f'{crosstab_col.name}_pattern']=pattern_list

  display(df_pattern_crosstab[[f'{crosstab_col.name}_pattern']])

  return df_pattern_crosstab[[f'{crosstab_col.name}_pattern']]
