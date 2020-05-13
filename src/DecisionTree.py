#! /usr/bin/env python3
"""
---2020.05.07--------------------------
# 决策树(Decision Tree)构成:
===============
1. 节点(Node)
    - 内部节点(Internal Node),内部节点可以看作是子树的根节点
    - 叶子节点(Leaf Node)
2. 有向边(Directed Edge)

# 决策树的数据结构
=================
数据结构:采用字典(Dict)的形式存放
Root(InternalNode)
|-  LeafNode
|-  InternalNode
    |- LeafNode

{InternalNode:{DirectedEdge:LeafNode,DirectedEdge:{InternalNode:{DirectedEdge:LeafNode}}}}
示例:(判断用户是否可注册网站账号)
{'是否年满18':{'Y':'可注册','N':{'是否有监护人':{'Y':'可注册','N':'不可注册'}}}}

# 决策数算法:ID3
=================
ID3只能用于解决离散内容的决策树
比如:
- 是否年满18(Y/N,2个选项), 温度(高/中/低,3个选项) 这些是可以使用ID3来生成决策树的.
- 收入金额(连续的值),身高(连续的值),类似这些是无法使用ID3来构建和生成决策树的,需要用到其他的算法.

>我们用一个例子来说明:根据下面2个条件来判断一个生物是否是鱼类
ID    离开水面可生存  有脚蹼    鱼类
1     Y             Y        Y
2     Y             Y        Y
3     Y             N        N
4     N             Y        N
5     N             Y        N

ID3算法的步骤:
1) 寻找当前作为InternalNode的最优特征
    - 计算每个特征的信息增益(Information Gain),信息增益最高的就是最优特征
2) 最优特征的特征值作为有向边,然后判断待分类结果的信息
    - 如果都是都是一类的话,输出叶子节点(Leaf Node)
    - 如果不是一类的话,对剩余的数据进行InternalNode最优解寻找
3) 直到所有数据被遍历完成或者,已经全部输出叶子节点,决策树构建完成

# 概率(Probability)
==================
假设现在有7个球: 4个篮球,3个足球,那么获取一个球,是足球的概率是多少?
P(足球) = 3/7, P(篮球) = 4/7

代码示例:
ball = np.array(['篮球','篮球','篮球','篮球','足球','足球','足球'])
proba = [ball[ball==val].size / ball.size for val in np.unique(ball)]

# 条件概率(Condition Probability)
================================
假设现在有7个球,4个篮球,3个足球,分别放在下面A,B框内.那么A框内取到足球的概率是多少?
A: 篮球2个,足球1个
B: 篮球2个,足球2个
P(足球|A) = 1/3  P(篮球|A) = 2/3
P(足球|B) = 2/4  P(篮球|B) = 2/4
这个就叫做条件概率,在已知A的情况下,足球/篮球的概率

代码示例:
ball = np.array([[2,1],[2,2]])

# 已知A或B的情况下, 求篮球或足球的条件概率
proba = ball / ball.sum(axis=1).reshape(-1,1)

# 已知篮球或足球的情况下,求A或B的条件概率
proba = ball / ball.sum(axis=0).reshape(1,-1)


# 熵(Entropy)
=============
熵也叫做香农熵(Shannon Entropy),用来表示随机变量的不确定性程度,值越高代表不确定程度越高.

计算公式:
H(x) = - np.sum(P* np.log2(P))
P是P(x)的概率数据集合,其总和为1. 比如[0.1,0.2,0.3,0.4]
用2为底的熵的单位叫做比特(bit),使用np.log2函数,在计算机科学中,我们一般使用这个.
用e为底的熵的单位叫做纳特(nat),使用np.log

代码示例:
>有一堆球,其中4个篮球,3个足球.这一堆球的熵为
ball = np.array([0,0,0,0,1,1,1])
lbl = np.unique(ball)
proba = [ball[ball==val].size / ball.size for val in lbl]
H = -np.sum(proba*np.log2(proba))

解释下:
这里最终的结果是0.98,非常接近1了,就代表从这堆球中拿出一个样本很难确定是足球还是篮球,因为大家的概率都接近0.5,
所以,它的不确定是非常高的.如果只有足球或这篮球,那么熵就是为0.拿出的要么就是篮球,要么就是足球,确定性很高


# 条件熵(Conditional Entropy)
============================
H(Y|X),条件熵代表在已知变量X的条件下,Y的不确定性.

计算公式:
H = np.sum(P*H(Y|X))
H(Y|X):在X的条件下,Y的熵
P:X的概率

代码示例:
>假设现在有7个球,分别放在下面0,1框内.
>0: 篮球2个,足球1个
>1: 篮球2个,足球2个
>根据足球和篮球的个数去判定是0框还是1框

ball = np.array([[2,1,0],[2,2,1]])
proba = lambda v: [v[val==v].size / v.size for val in np.unique(v)]
entropy = lambda x: -np.sum(x*np.log2(x))

这里计算足球的条件熵
>先获取足球这里的概率,分别是0.5, 0.5
p_football = proba(ball[:,1])

ball[ball[:,1]==1]

1*entropy(p_box)
0.5*0+0.5*0

# 信息增益(Information Gain)
===========================
g(D,A) = H(D)-H(D|A)


---参考资料----------------------------------
<统计学习方法> - 李航
<机器学习实战> - Peter Harrington
"""
import numpy as np
import pandas as pd

# no_surfacing = np.array(list('YYYNN'))
# flippers = np.array(list('11011'))
# fish = np.array(list('yynnn'))
# ds = np.c_[no_surfacing, flippers, fish]
# lbl = np.array('no_surfacing,flippers,fish'.split(','))

# df = pd.read_excel('~/code/jhml/ds.xlsx',sheet_name='dtree')
df = pd.read_excel('~/code/jhml/ds.xlsx',sheet_name='playTennis')
# df = pd.read_excel('~/code/jhml/ds.xlsx',sheet_name='fish')
ds = np.array(df)
lbl = np.array(df.columns)
dt = DecisionTree(ds, lbl)
dt.tree
dt.predict(ds,lbl)

class DecisionTree:
    def __init__(self, ds, lbl):
        self.tree = self.tree_build_id3(ds, lbl)

    def entropy_calc(self, x):
        proba = [x[x == tag].size / x.size for tag in np.unique(x)]
        return -np.sum(proba * np.log2(proba))

    def tree_build_id3(self, ds, lbl):
        node = {};node_val = {}
        cols = np.ones(lbl.size).astype(bool)

        # 计算信息增益Information Gain
        #base_ent = self.entropy_calc(ds.T[-1])
        conditional_ent = [
            np.sum([d[d == tag].size / d.size * self.entropy_calc(ds[tag == d].T[-1])
                    for tag in np.unique(d)])
            for d in ds.T[:-1]
        ]
        idx = np.array(conditional_ent).argmin()
        #info_gain = base_ent - np.array(conditional_ent)
        #idx = info_gain.argmax()
        # 构建内部节点Internal Node
        node[lbl[idx]] = node_val

        # 已经处理的特征移除
        cols[idx] = False

        # 构建有向边Directed Edge
        for val in np.unique(ds.T[idx]):
            flg = ds.T[idx] == val
            if abs(self.entropy_calc(ds[flg].T[-1])) < 1e-7:
                node_val[val] = ds[flg,-1][0]
            else:
                node_val[val] = self.tree_build_id3(ds[flg][:,cols],lbl[cols])
        return node

    def predict(self,X,lbl):
        if X.ndim == 1:
            ret = self.predict_one(X,lbl)
        else:
            ret = [self.predict_one(x,lbl) for x in X]
        return ret

    def predict_one(self,x,lbl):
        tree = self.tree
        while True:
            node = [v for v in tree.keys()][0]
            node_val = [v for v in tree.values()][0]
            ret = node_val[x[node == lbl][0]]
            if isinstance(ret,dict):
                tree = ret
            else:
                return ret
