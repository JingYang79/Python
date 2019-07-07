from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
#pca.txt是一个没有表头的多维数据，一共有7列，利用pandas读取
df = pd.read_table('d:/PCA.txt')
#将df转换成矩阵
dataMat = np.array(df)
#调用sklearn中的PCA，其中主成分有5列
pca_sk = PCA(n_components=5)
#利用PCA进行降维，数据存在newMat中
newMat = pca_sk.fit_transform(dataMat)
#利用KMeans进行聚类，分为3类
kmeans = KMeans(n_clusters=3,random_state=0).fit(newMat)
#labels为分类的标签
labels = kmeans.labels
#把标签加入到矩阵中用DataFrame生成新的df，index为类别的编号，这里是0,1,2
dataDf = pd.DataFrame(newMat,index=labels,columns=['x1','x2','x3','x4','x5'])
#数据保存在excel文件中
dataDf.to_excel('d:/pca_cluster.xls')
print(pca_sk.explained_variance_ratio_)


# https://blog.csdn.net/anneqiqi/article/details/64443711
# https://blog.codinglabs.org/articles/pca-tutorial.html
# https://blog.csdn.net/u010665216/article/details/78382984
