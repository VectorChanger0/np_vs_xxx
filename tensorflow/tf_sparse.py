import numpy as np
import tensorflow as tf

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)


def tf_sparse_tensor_to_dense(N0=3, N1=5):
    np1 = np.random.rand(N0, N1)
    np2 = np.random.rand(N0,N1)>0.5
    np1[np.logical_not(np2)] = 0

    tmp1 = np.stack(np.where(np2), axis=1)
    tf1 = tf.SparseTensor(tmp1, np1[np2], [N0,N1])
    tf2 = tf.sparse_tensor_to_dense(tf1)

    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('tf_sparse_tensor_to_dense:: np vs tf: ', hfe_r5(np1,tf2_))


if __name__=='__main__':
    tf_sparse_tensor_to_dense()
