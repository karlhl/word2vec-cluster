# word2vec 聚类

提供一个对于短文本，使用word2vec训练词向量，并对词向量进行聚类

![image-20210314193407595](https://gitee.com/karlhan/picgo/raw/master/img/image-20210314193407595.png)

word2vec+kmeans.py 文件，对一个时间段内进行词向量训练，然后根据tfidf提取的关键词进行kmeans聚类。

word2vec+kmeans2.py 文件，对所有隐患描述进行词向量训练，然后提取一个时间段内的关键词进行聚类。

word2vec+dbscan.py 文件,使用dbscan聚类。

show_figure.py 读取model，进行pca降维进行绘制图像

![image-20210314193847430](https://gitee.com/karlhan/picgo/raw/master/img/image-20210314193847430.png)

可能是数据原因，其实效果并不太好。

**测试**

使用腾讯的Tencent_AILab_ChineseEmbedding训练好的词向量进行聚类测试，并进行标签标注，这里仅用五个词，直接查询词向量文件。

![image-20210314201447970](https://gitee.com/karlhan/picgo/raw/master/img/image-20210314201447970.png)



**待改进**

有没有可能直接使用腾讯训练好的词向量，对已有短文本提取关键词聚类