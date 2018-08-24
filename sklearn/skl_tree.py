import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfe_r5 = lambda x,y,eps=1e-3: round(hfe(x,y,eps),5)


class _CopyTree(object):
    def __init__(self, sk_tree):
        self.children_left = sk_tree.children_left
        self.children_right = sk_tree.children_right
        self.feature = sk_tree.feature
        self.threshold = sk_tree.threshold
        self.value = sk_tree.value[:,0,0]
    def is_leaf(self, ind1):
        return (self.children_left[ind1]==-1) and (self.children_right[ind1]==-1)
    def transform_single(self, X, ind1):
        if self.is_leaf(ind1):
            return self.value[ind1]
        tmp1 = X[self.feature[ind1]] <= self.threshold[ind1]
        if tmp1:
            return self.transform_single(X, self.children_left[ind1])
        else:
            return self.transform_single(X, self.children_right[ind1])
    def transform(self, X):
        return np.array([self.transform_single(x,0) for x in X])


def skl_decision_tree_regr(N0=100, N1=4):
    pass
    np1 = np.random.rand(N0,N1)
    np2 = np.random.rand(N0)

    regr = DecisionTreeRegressor(max_depth=5)
    regr.fit(np1, np2)
    np3 = regr.predict(np1)

    myTree = _CopyTree(regr.tree_)
    np4 = myTree.transform(np1)
    print('skl_decision_tree_regr:: skl vsnp :', hfe_r5(np3,np4))


if __name__=='__main__':
    skl_decision_tree_regr()
