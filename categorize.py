import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display
from util import *

#二つのデータを比較したhist図を出力
#列名が異なる場合はskip
def plot_compare_hist_data(df1,df2,label1,label2):
    '''
    引数：
    df1、df2＝比較対象となるdf
    label1,label2=それぞれのdfにつけるラベル
    ーーーーーーーーー
    出力：
    横並びのhist
    '''
    cols_1=df1.columns
    cols_2=df2.columns

    for i in range(len(cols_1)):
        if cols_1[i]==cols_2[i]:
            plt.hist([df1[cols_1[i]],df2[cols_2[i]]],label=[label1,label2])
            plt.legend(loc="upper left")
            plt.title(cols_1[i])
            plt.xticks(rotation=90)
            plt.show()


#二つのデータを比較した箱ひげ図を出力
#列名が異なる場合,データに数値以外が含まれる場合はskip
def plot_compare_boxplot_data(df1,df2,label1,label2):
    '''
    引数：
    df1、df2＝比較対象となるdf
    label1,label2=それぞれのdfにつけるラベル
    ーーーーーーーーー
    出力：
    横並びの箱ひげ図
    '''
    df1=extract_int_float_data(df1)
    df2=extract_int_float_data(df2)
    cols_1=df1.columns
    cols_2=df2.columns

    for i in range(len(cols_1)):
        if cols_1[i]==cols_2[i]:
            points = (df1[cols_1[i]],df2[cols_2[i]])
            fig, ax = plt.subplots()
            bp = ax.boxplot(points) 
            ax.set_xticklabels([label1,label2])
            plt.title(cols_1[i])
            plt.show()
    
#KMeansによる分類を実行する
def exe_KMeans(df,n_clusters):
  '''
  df=df
  n_clusters=分類するクラスター数
  '''
  data=df.values
  model=KMeans(n_clusters=n_clusters)
  model.fit(data)

  return model.labels_