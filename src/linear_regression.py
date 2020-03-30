"""
---2020.03.27 Jian.Hu-------------------
1). 优化__intercept_fit函数成__input_fit函数,增加对1维输入的处理
2). 优化权重参数的设定函数,将权重参数的设定统一在fit函数,使得代码更加容易理解
3). 因权重参数的优化,而相对应优化的__mean_square_error函数
"""
import numpy as np

class LinearRegression:
    def __init__(self,fit_inter=True):
        self.__fit_inter = fit_inter

    def __input_fit(self,X):
        if self.__fit_inter: #对intercept的转换
            X = np.c_[np.ones(X.shape[0]),X]
        if X.ndim == 1: # 对1维输入的转换
            X = X.reshape(-1,1)
        return X

    def __normal_equation(self,X,y):
        X = self.__input_fit(X)
        return np.linalg.pinv(X.T @ X) @ X.T @ y

    def __mean_square_error(self,X,y,W):
        pred = self.__input_fit(X) @ W
        return np.sum((pred-y)**2)/(2*X.shape[0])

    def __gradient_descent(self,X,y,W,lr,itercnt):
        delta = 1e-7
        grad = np.zeros_like(W)

        for _ in range(itercnt):
            for i in range(grad.shape[0]):
                # 使用数值微分求导
                t = W[i]
                W[i] = t + delta
                f1 = self.__mean_square_error(X,y,W)
                W[i] = t - delta
                f2 = self.__mean_square_error(X,y,W)
                grad[i] = (f1-f2)/(2*delta)
                W[i] = t
            update = lr*grad

            update_absval = np.abs(update)
            if np.any(update_absval) > 1e7:  # 出现学习率过大的情况
                break
            if np.all(update_absval) < 1e-7: # 基本已经收敛,无须继续下降
                break

            # 同步更新参数
            W -= update
        return W

    def fit(self,X,y,W=None,lr=0.01,itercnt=1e5):
        # y,W都平铺,保证ndarray运算的shape是一致的
        y = y.flatten()
        if W is None:
            W = self.__normal_equation(X,y)
        else:
            W = W.flatten()
            itercnt = int(itercnt)
            W = self.__gradient_descent(X,y,W,lr,itercnt)

        # 权重系数设定
        self.W = W
        self.intercept_ = W[0] if self.__fit_inter else 0.
        self.coef_ = W[self.__fit_inter:]

    def predict(self,X):
        X = self.__input_fit(X)
        if not hasattr(self,'W'):
            raise Exception('You Need Fit Model First')
        return X @ self.W


if __name__ == '__main__':
    a = np.array([1,3])
    b = np.array([2,4])
    x = np.array([1,2,3,4]).reshape(2,2)
    y = 0.2+ 0.8 * a + 0.3 * b
    w = np.array([0.1,0.4,0.15])
    lr = LinearRegression()
    lr.fit(x,y,w)
    #lr.W
    lr.W
    lr.predict(x)
    lr.fit(a,y,w)
