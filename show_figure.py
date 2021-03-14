from gensim.models import Word2Vec
import gensim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
# model = Word2Vec.load('mymodel/word2vec2+pac.bin')
# wv = model.wv
#
# print(model.wv.vocab.keys())
# print(model.wv.vocab["堆码"])
# print(model.wv.index2word)
#
#
# pca = PCA(n_components=2)
# newData = pca.fit_transform(wv)   #载入N维
# print(newData)

from gensim.models import Word2Vec
from random import sample
from pylab import mpl
from sklearn.manifold import TSNE

model = Word2Vec.load('mymodel/word2vec+dbscan.bin')

df = pd.read_excel("word2vec+dbscan.xls",header = 0)
words = list(df["word"])
print(words)

label = list(df["label"])
print(label)

#载入模型，这个模型中的词向量是100维的，模型文件放在当前目录
# words = list(model.wv.vocab)
#把所有词向量的词语装到列表中
# labels = sample(words, 20) #随机取出1000个词语用来画图
tokens = model.wv.__getitem__(words) #得到1000个词语的词向量

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(tokens)
#降维处理
mpl.rcParams['font.sans-serif'] = ['SimHei'] #中文字体
mpl.rcParams['axes.unicode_minus'] = False #防止负号出现异常显示
x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])

plt.figure(figsize=(16, 16)) #定义画布大小
plt.scatter(x, y, c = label, s = 180, cmap = plt.cm.Spectral)

for i in range(len(x)):
    # plt.scatter(x[i],y[i],c=label[i],cmap=plt.cm.Spectral)
    plt.annotate(words[i],
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

plt.show()