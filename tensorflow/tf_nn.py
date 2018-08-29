import numpy as np
import tensorflow as tf

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)


def tf_dropout(N0=5, N1=1000, keep_prob=0.25):
    np1 = np.ones((N0,N1))

    tf1 = tf.constant(np1)
    tf2 = tf.nn.dropout(tf1, keep_prob=keep_prob)
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)

    tmp1 = np.abs(tf2_-0)<1e-5
    tmp2 = np.abs(tf2_-1/keep_prob)<1e-5
    tmp3 = np.all(np.logical_or(tmp1, tmp2))
    print('0 or 1/keep_prob: ', tmp3)
    tmp4 = '  '.join(['{:.3f}'.format(x) for x in tmp2.mean(axis=1)])
    print('keep rate along each batch: ', tmp4)


def tf_dropout01(N1=100, keep_prob=0.3):
    np1 = np.random.rand(N1)
    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1)
        tf2 = tf.nn.dropout(tf1, keep_prob, name='dropout01')
        tf3 = tfG.get_tensor_by_name('dropout01/random_uniform:0')
    with tf.Session(graph=tfG) as sess:
        tf2_,tf3_ = sess.run([tf2,tf3])
    np2 = np.floor(tf3_+keep_prob) * np1/keep_prob
    print('tf_dropout01:: np vs tf: ', hfe_r5(np2, tf2_))


def tf_top_k(N0=4, N1=10, k=3):
    np1 = np.random.rand(N0, N1)
    np2 = np.sort(np1, axis=1)[:,::-1][:,:k]
    np3 = np.argsort(np1, axis=1)[:,::-1][:,:k]

    tf1 = tf.constant(np1)
    tf2, tf3 = tf.nn.top_k(tf1, k)
    with tf.Session() as sess:
        tf2_,tf3_ = sess.run([tf2,tf3])

    print('top_k, index:: np vs tf: ', hfe_r5(np2, tf2_))
    print('top_k, value:: np vs tf: ', hfe_r5(np3, tf3_))


def tf_normalize_with_moments(N0=100, N1=5):
    np1 = np.random.rand(N0, N1)
    np2 = (np1-np1.mean(axis=0, keepdims=True))/np1.std(axis=0, keepdims=True)

    tf1 = tf.constant(np1)
    mean, var = tf.nn.moments(tf1, axes=[0], keep_dims=True)
    tf2 = (tf1-mean)/tf.sqrt(var)
    
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('normalize:: tf vs np: ', hfe_r5(np2, tf2_))


if __name__=='__main__':
    tf_dropout()
    print('')
    tf_dropout01()
    print('')
    tf_top_k()
    print('')
    tf_normalize_with_moments()
