# KNN

KNN(K-Nearest Neighbor),K近邻算法.它属于监督学习,可用于分类(Classification)和回归(Regression)问题.

KNN算法的3个要素

- 距离度量
- K值选择
- 分类决策



# 距离度量

计算公式:  Lp距离(Lp Distance, 也被称为minkowski distance)
$$
L_p(x_i,x_j) = (\sum_{l=1}^n |x_i^{(l)}-x_j^{(l)}|^{p})^{\frac{1}{p}}
\\
其中,x_i,x_j \in X , X是输入数据的特征空间, p值要\ge 1,
$$

- p=2的时候,Lp距离被称为欧式距离(Euclidean Distance) , **默认距离度量都采用欧式距离**
- p=1的时候,Lp距离被称为曼哈顿距离(Manhattan Distance)

> 举例说明

下表是鸢(yuan)尾花的数据(Setora,Versicolor,Virginca代表的是不同种类的鸢尾花), 

它记录了鸢尾花的 花萼(Sepal),花瓣(Petal) 的长度和宽度

| Sepal Lenth(cm) | Sepal Width(cm) | Petal Length(cm) | Petal Width(cm) |      Target |
| --------------: | --------------: | ---------------: | --------------: | ----------: |
|             5.1 |             3.5 |              1.4 |             0.2 |      Setosa |
|             5.9 |             3.2 |              4.8 |             1.8 |  Versicolor |
|             6.5 |             3.4 |              5.6 |             2.4 |    Virginca |
|             5.0 |             3.4 |              1.6 |             0.4 | T.B.D(待定) |

欧式距离的计算,以最后一行数据和第一行数据为列:
$$
Lp = (|5.0-5.1|^2 + |3.4-3.5|^2 + |1.6-1.4|^2+|0.4-0.2|^2)^{\frac{1}{2}}
\\
等价于
\\
Lp = \sqrt{|5.0-5.1|^2 + |3.4-3.5|^2 + |1.6-1.4|^2+|0.4-0.2|^2}
$$


# K值选择

K值代表我要选取最近的K个距离点,

![K值选择](https://github.com/joestarhu/jhml/blob/master/knn/knn-K%E5%80%BC%E9%80%89%E6%8B%A9.png?raw=true)



# 分类决策



# KD-Tree

## 构建



## 搜索

