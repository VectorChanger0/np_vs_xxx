import random
import numpy as np
import numpy.linalg
from sklearn.decomposition import PCA

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfe_r5 = lambda x,y,eps=1e-3: round(hfe(x,y,eps),5)

def skl_pca(N0=1000, N1=10):
    np1 = np.random.rand(N0, N1)
    n_components = random.randint(3, np1.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(np1)
    np2 = pca.transform(np1)

    tmp1 = np1 - np1.mean(axis=0)
    tmp2 = np.matmul(tmp1.transpose(1,0), tmp1)/(tmp1.shape[0]-1)
    _,S,V = np.linalg.svd(tmp2)

    tmp1 = hfe_r5(pca.explained_variance_, S[:n_components])
    print('skl_pca_explained_variance:: np vs skl: ', tmp1)

    tmp1 = hfe_r5(pca.explained_variance_ratio_,S[:n_components]/S.sum())
    print('skl_pca_explained_variance_ratio:: np vs skl: ', tmp1)

    tmp1 = np.sum(V[:n_components,:] * pca.components_, axis=1)
    print('skl_pca_components:: np vs skl: ', hfe_r5(np.abs(tmp1), 1))

    tmp1 = hfe_r5(np2, np.matmul(np1-np1.mean(axis=0), pca.components_.transpose((1,0))))
    print('skl_pca_transform:: np vs skl: ', tmp1)

if __name__=='__main__':
    skl_pca()
