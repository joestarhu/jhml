#! /usr/bin/env python3
"""
---2020.04.08---------------------------------------------
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
        A:np.ndarray,shape(m,n),输入数据
        B:np.ndarray,shape(t,n),待分类数据
        p是变量,取值>=1

    其中当p=1的时候,Manhattan Distance(曼哈顿距离)
    L1 = np.sum(np.abs(A-B),axis=1) ** (1/1)

    其中当p=2的时候,Euclidean Distance(欧氏距离)
    L2 = np.sum(np.abs(A-B)**2, axis=1) ** (1/2)

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

class kNN:
    def predict(self,X,y,T,k=3,normalize=False,lp='l2'):
        """
        通过kNN算法预测分类

        Parameters:
        -----------
        X: np.ndarray 输入训练数组
        y: np.ndarray 输入训练标签
        T: np.ndarray 输入预测数据
        k: int k值
        normalize: bool 正规化
        lp: str lp参数选择 可选内容为[l1,l2]

        Returns:
        ----------
        np.ndarray 预测数据的分类
        """
        if normalize:
            ds = np.r_[X,T]
            max,min = ds.max(axis=0),ds.min(axis=0)
            ds = (ds-min)/(max-min)
            X = ds[:X.shape[0]]
            T = ds[X.shape[0]:]

        p = 2 if lp == 'l2' else 1
        return np.array([self.__getNearestLabel(X,y,val,k,p) for val in T])

    def __getNearestLabel(self,X,y,t,k,p):
        # 计算lp Distance
        lp = np.sum(np.abs(t - X) ** p, axis=1) ** (1/p)
        # 获取最近(数值最小的)k个标签
        nlbl = y[lp.argsort()[:k]]
        # 从标签中获取值最大的一个,作为输出
        d = {}
        for i in nlbl:
            d[i] = d.get(i,0) + 1
        return sorted(d,reverse=True)[0]

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt
    iris = load_iris()

    X = np.r_[iris.data[:30],iris.data[50:80],iris.data[100:130]]
    y = np.r_[iris.target[:30],iris.target[50:80],iris.target[100:130]]
    tX = np.r_[iris.data[30:50],iris.data[80:100],iris.data[130:]]
    ty = np.r_[iris.target[30:50],iris.target[80:100],iris.target[130:]]
    
    ret = np.array([ty[kNN().predict(X,y,tX,k=i,normalize=True)==ty].size/ty.size for i in range(3,20)])
    print(ret)
