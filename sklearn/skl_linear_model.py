import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso

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


def skl_ridge(N0=8, N1=9, noise=0.3, alpha=np.logspace(-10,-2,100), use_meaningless_data=True):
    '''http://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html'''
    if use_meaningless_data:
        print('WARNING! the used data CANNOT explain the effect of alpha.')
        npx = 1 / (np.arange(0,N0)[:,np.newaxis] + np.arange(1,N1))
        npy = np.ones(N0)
    else:
        tmp1 = np.random.normal(0, 1, size=[N0,N1])
        tmp2 = np.random.normal(N1)
        npx = tmp1 + np.random.normal(0, noise/2, size=[N0,N1])
        npy = np.sum(tmp1*tmp2, axis=1)

    coefs = np.zeros([alpha.shape[0], npx.shape[1]])
    for ind1,a in enumerate(alpha):
        ridge = Ridge(alpha=a, fit_intercept=False)
        ridge.fit(npx, npy)
        coefs[ind1] = ridge.coef_

    fig = plt.figure()
    ax = fig.add_axes([0.15,0.15,0.7,0.7])
    ax.plot(alpha, coefs)
    ax.set(xscale='log', xlabel='alpha', ylabel='weights', title='skl_ridge')
    fig.show()


def skl_logistic_regression():
    # http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic.html#sphx-glr-auto-examples-linear-model-plot-logistic-py
    raise Exception('not implement yet, maybe one day')


if __name__=='__main__':
    skl_linear_regression()
    print('')
    skl_ridge()



from scipy import sparse
from scipy import ndimage


def _weights(x, dx=1, orig=0):
    x = np.ravel(x)
    floor_x = np.floor((x - orig) / dx).astype(np.int64)
    alpha = (x - orig - floor_x * dx) / dx
    return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))


def _generate_center_coordinates(l_x):
    X, Y = np.mgrid[:l_x, :l_x].astype(np.float64)
    center = l_x / 2.
    X += 0.5 - center
    Y += 0.5 - center
    return X, Y


def build_projection_operator(l_x, n_dir):
    """ Compute the tomography design matrix.
    l_x(int): linear size of image array
    n_dir(int): number of angles at which projections are acquired.
    (ret)p(n_dir, l_x, l_x**2): sparse matrix of shape
    """
    X, Y = _generate_center_coordinates(l_x)
    angles = np.linspace(0, np.pi, n_dir, endpoint=False)
    data_inds, weights, camera_inds = [], [], []
    data_unravel_indices = np.hstack((np.arange(l_x**2),np.arange(l_x**2)))
    for i, angle in enumerate(angles):
        Xrot = np.cos(angle) * X - np.sin(angle) * Y
        inds, w = _weights(Xrot, dx=1, orig=X.min())
        mask = np.logical_and(inds >= 0, inds < l_x)
        weights += list(w[mask])
        camera_inds += list(inds[mask] + i * l_x)
        data_inds += list(data_unravel_indices[mask])
    proj_operator = sparse.coo_matrix((weights, (camera_inds, data_inds)))
    return proj_operator


def generate_synthetic_data():
    """ Synthetic binary data """
    rs = np.random.RandomState(0)
    n_pts = 36
    x, y = np.ogrid[0:l, 0:l]
    mask_outer = (x - l / 2.) ** 2 + (y - l / 2.) ** 2 < (l / 2.) ** 2
    mask = np.zeros((l, l))
    points = l * rs.rand(2, n_pts)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndimage.gaussian_filter(mask, sigma=l / n_pts)
    res = np.logical_and(mask > mask.mean(), mask_outer)
    return np.logical_xor(res, ndimage.binary_erosion(res))


# Generate synthetic images, and projections
l = 128
proj_operator = build_projection_operator(l, l // 7)
data = generate_synthetic_data()
proj = proj_operator * data.ravel()[:, np.newaxis]
proj += 0.15 * np.random.randn(*proj.shape)

# Reconstruction with L2 (Ridge) penalization
rgr_ridge = Ridge(alpha=0.2)
rgr_ridge.fit(proj_operator, proj.ravel())
rec_l2 = rgr_ridge.coef_.reshape(l, l)

# Reconstruction with L1 (Lasso) penalization
# the best value of alpha was determined using cross validation
# with LassoCV
rgr_lasso = Lasso(alpha=0.001)
rgr_lasso.fit(proj_operator, proj.ravel())
rec_l1 = rgr_lasso.coef_.reshape(l, l)

plt.figure(figsize=(8, 3.3))
plt.subplot(131)
plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('original image')
plt.subplot(132)
plt.imshow(rec_l2, cmap=plt.cm.gray, interpolation='nearest')
plt.title('L2 penalization')
plt.axis('off')
plt.subplot(133)
plt.imshow(rec_l1, cmap=plt.cm.gray, interpolation='nearest')
plt.title('L1 penalization')
plt.axis('off')

plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                    right=1)

plt.show()