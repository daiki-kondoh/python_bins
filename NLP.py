#textを逆順に抽出する
def inverse_text(text):
  inverse_text=text[::-1]
  
  return inverse_text


#textから、文字をstep飛ばしで抽出する
def extraction_char_by_step(text,start,end,step):
  extraction_text=text[start:end:step]
  
  return extraction_text

#word_listから、文字をstep飛ばしで抽出する
def extract_char_from_word_list(word_list,start,end,step):
  extract_char_list=[]
  for i,v in enumerate(word_list):
    extract_char_list.append(v[start:end:step])
    
  return extract_char_list

#text1とtext2を交互に混ぜ合わせる
def fusion_text(text1,text2):
  fusion_text_list=[]

  for x,y in zip(text1,text2):
    fusion_text_list.append(x)
    fusion_text_list.append(y)
  
  fusion_text="".join(fusion_text_list)

  return fusion_text


#textを空白ごとに分解し、文字のリストを返す
def split_text_by_word(text):
  table = str.maketrans({
    ',': '',
    '.': '',
    '・': '',
    '。':'',
    '、':''
    })
  text_remove_comma=text.translate(table)
  word_list=[]
  for i in text_remove_comma.split():
    word_list.append(i)
  
  return word_list

#listに格納された文字のそれぞれの長さを返す
def count_word_len(word_list):
  count_word_len_list=[len(i) for i in word_list]
  return count_word_len_list


#listから指定した箇所のvalueを抽出する
def extract_value_from_list(lis,extract_num_list):
  extract_list=[]
  for i,n in enumerate(extract_num_list):
    extract_list.append(lis[n])

  return extract_list


#listの値に定数intを加算する
def list_plus_int(lis,int):
  eq=lambda x:x+int
  plus_list=list(map(eq,lis))
  
  return plus_list


#二つのlistからdictを作成する
def lists_to_dict(key_list,value_list):
  dict_from_list = dict(zip(key_list, value_list))

  return dict_from_list


#dictを結合する
def concat_dict(dict1,dict2):
  concated_dict=dict(**one_char_dict,**two_char_list)

  return concated_dict


#valueを基準にdictをソートする
def sort_dict_by_value(dic):
  sorted_list_from_dict=sorted(dic.items(),key=lambda value:value[1])
  sorted_dict=dict( (key,value) for key,value in sorted_list_from_dict)

  return sorted_dict


#textからngramを作成する
def create_ngram(text,n):
  ngram_list=[]
  for i in range(len(text)-n+1):
    ngram_list.append(text[i:i+n])

  return ngram_list


#二つのlistから和集合、積集合、差集合をそれぞれ計算する
def set_operation_list(list1,list2,operation):
  list1_set=set(list1)
  list2_set=set(list2)

  if operation=='or' or operation=='|':
    set_operation_list= list1_set.union(list2_set)
  elif operation=='and' or operation=='&':
    set_operation_list= list1_set.intersection(list2_set)
  elif operation=='difference' or operation=='-':
    set_operation_list= list1_set.difference(list2_set)

  return set_operation_list


#listの中に、指定したvalueが含まれているかを判定する
def judge_contain_value_in_list(value,lis):
  if value in lis:
    return f'{value}はリスト内に含まれています。'
  else:
    return f'{value}はリスト内に含まれていません。'

#textをランダムな順番に入れ替えて返す
import random
def rondom_sort_char(text):
  rondom_sorted_char=''.join(random.sample(text[::],len(text)))
  return rondom_sorted_char


#!apt install aptitude
#!aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y
#!pip install mecab-python3==0.7

#textに対して、形態素解析を実行する
import MeCab
def exe_mecab(text):

  for i in text.split('\n'):
      #形態素解析の実行
      mecab = MeCab.Tagger('mecab-ipadic-neologd').parse(i)
      print(i)
      print(mecab)
      #用意したm_listに格納します。
      mecab_list=mecab.split('\n')
 
  return mecab_list


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
