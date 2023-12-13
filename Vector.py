#dfの指定した列をベクトルとみなしてコサイン類似度を計算し、その類似度行列をdfで返す関数
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

def cul_cos_similarity_matrix(df,cols,id):
  ss = preprocessing.StandardScaler()
  standarized_df = ss.fit_transform(df[cols].fillna(0))
  cos_similarity_df = pd.DataFrame(cosine_similarity(standarized_df))
  cos_similarity_df=cos_similarity_df.set_index(df['ID'])
  cos_similarity_df.columns=df[id]

  return cos_similarity_df


#距離行列が格納されたdfに対してMDSを行う
from sklearn.manifold import MDS
#precomputed_df=距離行列のdf
#n_components=次元数
def cul_MDS_precomputed(precomputed_df,n_components):
  data_array=precomputed_df.values
  mds=MDS(n_components,metric=True,dissimilarity='precomputed')
  mds_coordinate_array=mds.fit_transform(data_array)

  return mds_coordinate_array


#距離行列が格納されたdfに対して２次元のMDSを行い、結果をplotする
import matplotlib.pyplot as plt
#precomputed_df=距離行列のdf
def MDS_scatter(precoputed_df):
  x=cul_MDS_precomputed(precoputed_df,2)
  plt.scatter(x[:,0],x[:,1])

  source=pd.DataFrame({'x':x[:,0],'y':x[:,1],'label':precoputed_df.index})
  points=alt.Chart(source).mark_point().encode(x='x:Q',y='y:Q')
  text=points.mark_text(align='left',baseline='middle',dx=7).encode(text='label')

  return points+text


#KMeansによる分類を実行する
#df=df
#n_clusters=分類するクラスター数
def exe_KMeans(df,n_clusters):
  data=df.values
  model=KMeans(n_clusters=n_clusters)
  model.fit(data)

  return model.labels_