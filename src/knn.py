"""
---2020.03.27 Jian.Hu---------------------------------------------
[算法简介]:
k-Nearest Neighbor(kNN) K近邻 ,用于监督学习的分类问题(可多分类)
简单来说就是:找到与目标值距离最近的k个元素,k个元素中最多的分类,就是目标值的分类.

kNN3大重要要素:
1). 距离度量
    使用: Lp Distance(也叫Minkowski Distance)
    它的公式为:
    Lp = np.sum(np.abs(A-B)**p,axis=1) ** (1/p)
    -------------
    ※其中:
        A为输入标签数据,shape为(m,n)
        B为待分类数据,shape为(m,1) 
        p是变量,取值>=1

    其中当p=1的时候,Manhattan Distance(曼哈顿距离)
    L1 = np.sum(np.abs(A-B))

    其中当p=2的时候,Euclidean Distance(欧式距离)
    L2 = np.sum((A-B) **2 ) ** (1/2)

2). k值选择(调参)
    k值的选择对算法会产生重大影响
    k值选择较小:
        会出现过拟合现象,即一些噪声点会影响到最终的决策
    k值选择较大:
        会出现欠拟合现象,即一些不相关的点会影响到最终的决策
    如何获取合适的k值:
        一般从一个较小的k值开始,交叉验证取到合适的k值

3). 决策标准
    多数表决,即k个近邻中多数的类决定待分类的数据

kNN的一些问题:
1). 计算量大,要计算所有输入样本的距离,耗时也耗空间
2). 基于输入数据来计算,无法给出数据的基础结构信息,无法获取到模型

---参考资料----------------------------------
<统计学习方法> - 李航
<机器学习实战> - Peter Harrington
"""
import numpy as np

class KNN:
    def classify(self,X:np.ndarray,y:np.ndarray,T:np.ndarray,k=3,norm=False,lp='l2'):
        d = {val:id for id,val in enumerate(['l1','l2'],start=1)}
        if lp not in d:
            raise Exception(f'lp value [l1,l2]')
        p = d[lp]

        # 正规化
        if True == norm:
            max = np.r_[X,T].max(axis=0)
            min = np.r_[X,T].min(axis=0)
            X = (X-min)/(max-min)
            T = (T-min)/(max-min)

        res = np.zeros(T.shape[0])
        for i in range(T.shape[0]):
            res[i] = self.__knn(X,y,T[i],k,p)
        return res

    def __knn(self,X,y,t,k,p):
        # 距离度量计算
        lp = np.sum(np.abs(t-X)**p,axis=1)**(1/p)
        # 获取k近邻的标签
        nearest = y[lp.argsort()[:k]]
        d = {}
        for i in nearest:
            d[i] = d.get(i,0) + 1
        return sorted(d,reverse=True)[0]

if __name__ == '__main__':
    X = np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])
    y = np.array([1,1,1,0,0,0])
    z = np.array([[18,92],[92,17],[101,23]])


    KNN().classify(X,y,z)
    KNN().classify(X,y,z,norm=True)
