"""
对时间段内的隐患 首先经过tf-idf进行关键词提取，每个主题保留五个关键词  再word2vec,然后进行k-mean聚类，结果存在word_closter2.xlsx中
其中k-mean是需要制定分几类的，效果不好
"""
import re, numpy as np, pandas as pd, jieba
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import jieba.analyse as anls
from sklearn.decomposition import PCA


def word_cluster(texts, n_clusters=3):
    # 分词
    texts = [[word for word in jieba.cut(sentence) if re.fullmatch('[a-zA-Z\u4e00-\u9fa5]+', word)]
             for text in texts for sentence in re.split('[\n。…；;!！?？]+', text)]
    # 词向量
    model = Word2Vec(texts,size=100,sg=0,iter=5,min_count=1)
    model.save("mymodel/word2vec+kmeans.bin")
    wv = model.wv
    print(type(wv))
    # 转为单位向量
    uv = np.linalg.norm(wv.vectors, axis=1).reshape(-1, 1)  # Unit Vector
    wv.vectors = wv.vectors / uv  # Vector or matrix norm
    # 聚类
    labels = KMeans(n_clusters).fit(wv.vectors).labels_
    # 输出excel
    df = pd.DataFrame([(w, labels[e]) for e, w in enumerate(wv.index2word)], columns=['word', 'label'])
    df.sort_values(by='label', inplace=True)
    df.to_excel('word2vec+kmeans.xls', index=False)


def load_data(path,time_window_start,time_window_end,use_tfidf):
    df = pd.read_excel(path,header = 0)
    df["时间"] = pd.to_datetime(df["时间"])
    df = df.set_index('时间')
    df.sort_values(by='时间', inplace=True)

    df1 =  df[time_window_start:time_window_end]

    # jieba分词添加停用词
    # jieba.analyse.set_stop_words("data/stop_words.txt")
    jieba.load_userdict("data/my_dict.txt")

    time_window = [] # 存储一定时间长度内的句子

    if use_tfidf:
        for i,r in df1.iterrows():
            for x,w in anls.extract_tags(str(r["隐患描述"]), topK=7, withWeight=True):
                time_window.append(x)
    else:
        for i,r in  df1.iterrows():
            time_window.append(' '.join(jieba.cut(r["隐患描述"])))
    return time_window





if __name__ == "__main__":
    # 获取时间窗内的隐患描述的内容列表
    time_window_start = "2018-04"  # 定义时间长度 '2016':'2017'
    time_window_end = "2019-07" # 设定开始结束时间
    time_window = load_data("data/data.xls",time_window_start,time_window_end,False)
    time_window = list(set(time_window))
    print(time_window)

    # 进行聚类，包括了分词，word2vec，k-means
    word_cluster(time_window)


    model = Word2Vec.load('mymodel/word2vec+kmeans.bin')
    print("---")
    print(model.wv.__getitem__("电梯"))