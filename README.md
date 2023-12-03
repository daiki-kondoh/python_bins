# util.py

# util.py

#各列の欠損値を表示する関数
def display_null_number(df):
  null_df=pd.DataFrame(df.isnull().sum().sort_values(ascending=False)).copy()
  null_df['null数']=null_df[0]
  display(null_df[['null数']][null_df['null数']>0])



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
