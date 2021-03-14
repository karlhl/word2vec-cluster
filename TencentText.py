"""
使用腾讯Tencent_AILab_ChineseEmbedding 测试聚类
"""

from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
# 从磁盘加载词向量文件
from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format("E:\\Tencent_AILab_ChineseEmbedding.txt",limit=10000,encoding="utf-8")

texts = ["猫","狗","自行车","喜欢","吃饭","汽车"]

wv = word_vectors.wv.__getitem__(texts)
print(type(wv))
# 转为单位向量
uv = np.linalg.norm(wv, axis=1).reshape(-1, 1)  # Unit Vector
wv = wv / uv  # Vector or matrix norm
# 聚类
labels = KMeans(3).fit(wv).labels_

for k,v in zip(texts,labels):
    print(k,v)