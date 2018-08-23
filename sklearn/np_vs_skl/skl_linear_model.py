import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, LogisticRegression

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfe_r5 = lambda x,y,eps=1e-3: round(hfe(x,y,eps),5)

def skl_linear_regression():
    tmp1 = load_boston()
    np1 = tmp1['data']
    np2 = tmp1['target']
    regr = LinearRegression()

    regr.fit(np1, np2)
    np3 = regr.predict(np1)
    np3_ = np.matmul(np1, regr.coef_) + regr.intercept_
    print('skl_linear_regression:: np vs skl: ', hfe_r5(np3,np3_))

def skl_logistic_regression():
    # http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic.html#sphx-glr-auto-examples-linear-model-plot-logistic-py
    raise Exception('not implement yet, maybe one day')


if __name__=='__main__':
    skl_linear_regression()
