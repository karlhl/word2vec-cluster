"""
训练model的时候使用所有文件，
再提取关键词，从model里查找词向量进行聚类

对时间段内的隐患 首先经过tf-idf进行关键词提取，每个主题保留五个关键词  再word2vec,然后进行k-mean聚类，结果存在word_closter2.xlsx中
其中k-mean是需要制定分几类的，效果不好
"""
import re, numpy as np, pandas as pd, jieba
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import jieba.analyse as anls
from sklearn.decomposition import PCA
import operator
from functools import reduce

def train_model(texts):
    model = Word2Vec(texts, size=100, sg=0, iter=5,min_count=1)
    model.save("mymodel/word2vec+kmeans2.bin")
    return model


def word_cluster(texts, n_clusters,model):
    # 分词
    texts = [[word for word in jieba.cut(sentence) if re.fullmatch('[a-zA-Z\u4e00-\u9fa5]+', word)]
             for text in texts for sentence in re.split('[\n。…；;!！?？]+', text)]
    # 词向量
    """
    这里有问题，不知道怎么转化
    """
    texts = reduce(operator.add, texts)
    print("texts",texts)

    wv = model.wv
    print(model["通道"])
    key_words_vectors = model.wv.__getitem__(texts)

    index2word = {i : c for i, c in enumerate(texts)}

    print("wv.vectors的shape",wv.vectors.shape)
    print("key_words_vectors的shape",key_words_vectors.shape)
    # 转为单位向量
    uv = np.linalg.norm(key_words_vectors, axis=1).reshape(-1, 1)  # Unit Vector
    key_words_vectors = key_words_vectors / uv  # Vector or matrix norm
    # 聚类
    labels = KMeans(n_clusters).fit(key_words_vectors).labels_
    # 输出excel
    # df = pd.DataFrame([(w, labels[e]) for e, w in enumerate(wv.index2word)], columns=['word', 'label'])
    df = pd.DataFrame([(w,labels[e]) for e,w in enumerate(texts)] , columns=['word', 'label'])
    df.sort_values(by='label', inplace=True)
    df.to_excel('word2vec+kmeans2.xls', index=False)


def load_data(path,time_window_start,time_window_end,use_tfidf):
    df = pd.read_excel(path,header = 0)
    df["时间"] = pd.to_datetime(df["时间"])
    df = df.set_index('时间')
    df.sort_values(by='时间', inplace=True)

    df1 =  df[time_window_start:time_window_end]

    # jieba分词添加停用词
    jieba.analyse.set_stop_words("data/stop_words.txt")
    jieba.load_userdict("data/my_dict.txt")

    time_window = [] # 存储一定时间长度内的句子

    if use_tfidf:
        for i,r in df1.iterrows():
            for x,w in anls.extract_tags(str(r["隐患描述"]), topK=5, withWeight=True):
                time_window.append(x)
    else:
        for i,r in  df1.iterrows():
            time_window.append(' '.join(jieba.cut(r["隐患描述"])))
    return time_window

def load_data_full(path):
    """
    获取所有时间内的隐患描述，用于训练model
    :param path:
    :return:
    """
    df = pd.read_excel(path, header=0)
    df["时间"] = pd.to_datetime(df["时间"])
    df = df.set_index('时间')
    df.sort_values(by='时间', inplace=True)

    # jieba分词添加停用词
    jieba.analyse.set_stop_words("data/stop_words.txt")
    jieba.load_userdict("data/my_dict.txt")

    time_window_full = []  # 存储一定时间长度内的句子

    for i, r in df.iterrows():
        temp = '/'.join(jieba.cut(r["隐患描述"])).replace('，','').replace('、','').split('/')
        # print(temp)
        time_window_full.append(temp)
    return time_window_full



if __name__ == "__main__":
    # 获取时间窗内关键词的隐患描述的内容列表
    time_window_start = "2018-04"  # 定义时间长度 '2016':'2017'
    time_window_end = "2018-04" # 设定开始结束时间
    data_path = "data/data.xls" # 原始数据路径
    # 获取一个时间段内的数据的关键词
    time_window = load_data(data_path,time_window_start,time_window_end,True) # 一维
    time_window = list(set(time_window))

    print("time_window",time_window)

    # 利用所有标题训练model
    time_window_full = load_data_full(data_path) # time_window_full是一个二维list
    print("time_window_full",time_window_full)
    model = train_model(time_window_full)
    # print("词库所有词",model.wv.vocab)


    # 进行聚类，包括了分词，word2vec，k-means
    n_clusters = 3
    word_cluster(time_window,n_clusters,model)




    model = Word2Vec.load('mymodel/word2vec+kmeans2.bin')
    print("---")
    # print(model.wv.__getitem__("风险"))