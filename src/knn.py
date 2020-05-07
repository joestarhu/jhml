#! /usr/bin/env python3
"""
---2020.05.07---------------------------------------------
1). 优化KNN算法,代码缩减到5行
2). 优化输出结果,输出结果包含分类结果和分类概率

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
import pandas as pd


class KNN:
    @staticmethod
    def predict(X,y,T,k=3,p=2,normalize=False) -> pd.DataFrame:
        """
        KNN, 采用Brute方式(计算目标和样本的每个距离)

        Paramters:
        ----------------
        X: np.ndarray
            训练数据集(Training Datasets)
        y: np.ndarray
            训练数据集标签(Training Datasets Label)
        T: np.ndarray
            测试数据集(Testing Datasets)
        k: int
            K值,Default:3
        p: int
            P值,Default:2
            P值为1就是曼哈顿距离(Mahattan Distance)
            P值为2就是欧式距离(Euclidean Distance)
        normalize: bool
            正规化(归一化)标签,Default:False
            True:   实施归一化
            False:  对输入数据不做处理

        Returns:
        ---------
        pd.DataFrame

        示列:
            动作	       爱情	     CLS
        0	0.333333	0.666667	爱情
        1	1.000000	0.000000	动作
        """
        if normalize:
            ds = np.r_[X,T]
            ds_min,ds_max = ds.min(axis=0),ds.max(axis=0)
            ds = (ds - ds_min)/(ds_max-ds_min)
            X = ds[:X.shape[0]]
            T = ds[X.shape[0]:]

        lbl = np.unique(y)
        n_lp = [np.sum(np.abs(t-X)**p,axis=1)**(1/p) for t in T]
        n_idx = np.argsort(n_lp,axis=1)[:,:k]
        proba = [[n[n==v].size / n.size for v in lbl] for n in y[n_idx]]
        df = pd.DataFrame(proba,columns=lbl)
        df['CLS'] = lbl[np.argmax(proba,axis=1)]
        return df

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt
    iris = load_iris()

    X = np.r_[iris.data[:30],iris.data[50:80],iris.data[100:130]]
    y = np.r_[iris.target[:30],iris.target[50:80],iris.target[100:130]]
    tX = np.r_[iris.data[30:50],iris.data[80:100],iris.data[130:]]
    ty = np.r_[iris.target[30:50],iris.target[80:100],iris.target[130:]]

    for k in range(3,20,2):
        error_size = ty.size - ty[KNN.predict(X,y,tX,k=k)['CLS'] == ty].size
        print(1 - error_size/ty.size)
