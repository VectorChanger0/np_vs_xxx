import numpy as np
import tensorflow as tf
from itertools import zip_longest

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)


def tf_count01234(N0=4):
    np1 = np.sort(np.random.randint(N0, size=(20,)))
    np2 = (np1==np.arange(N0)[:,np.newaxis]).sum(axis=1)

    ret_np = np.zeros((N0,))
    for ind1 in range(N0):
        ret_np[ind1] = (np1==ind1).sum()

    tf1 = tf.constant(np1, dtype=tf.int32)
    tf2 = tf.math.reduce_sum(tf.cast(tf.equal(tf1, tf.range(N0)[:,tf.newaxis]),tf.int32), axis=1)
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('tf_count01234:: np vs tf: ', hfe(np2,tf2_))


def tf_boolean_mask(N0=5, N1=7):
    np1 = np.random.rand(N0, N1)
    np2 = np.random.rand(N0, N1)>0.5
    np3 = np1[np2]

    tf1 = tf.constant(np1)
    tf2 = tf.constant(np2, dtype=tf.bool)
    tf3 = tf.boolean_mask(tf1, tf2)
    with tf.Session() as sess:
        tf3_ = sess.run(tf3)
    print('boolean_mask:: tf vs np: ', hfe_r5(np3, tf3_))


def tf_map_fn_sequence_boolean_mask():
    np1 = np.random.rand(10, 20)
    np2 = np.random.randint(low=5, high=15, size=[10])
    np3 = np.concatenate([x[:y]*np.mean(x[y:]) for x,y in zip(np1,np2)])

    tf1 = tf.constant(np1, dtype=tf.float32)
    tf2 = tf.constant(np2, dtype=tf.int32)
    def hf1(xy,range_max=15):
        x,y = xy
        tmp1 = x[:y]*tf.math.reduce_mean(x[y:])
        ret1 = tf.shape(tmp1)[0]
        ret2 = tf.pad(tmp1,[[0,range_max-ret1]])
        ret2.set_shape([range_max])
        return ret1,ret2
    tf3,tf4 = tf.map_fn(hf1, [tf1,tf2], dtype=(tf.int32,tf.float32))
    tf5 = tf.boolean_mask(tf4, tf.sequence_mask(tf3,15))
    with tf.Session() as sess:
        tf5_ = sess.run(tf5)
    print('map_fn sequence boolean mask:: np vs tf: ', hfe_r5(np3, tf5_))


def tf_scatter_nd(N1=20):
    num1 = int(round(N1*0.3))
    np1 = np.random.permutation(N1)[:num1]
    np2 = np.random.rand(num1)
    np3 = np.zeros(N1)
    np3[np1] = np2

    tf1 = tf.constant(np1)
    tf2 = tf.constant(np2)
    tf3 = tf.scatter_nd(tf1[:,tf.newaxis], tf2, (N1,))
    with tf.Session() as sess:
        tf3_ = sess.run(tf3)
    print('scatter_nd advanced assign:: np vs tf: ', hfe_r5(np3, tf3_))


def tf_embedding_lookup(max_int=1000, embedding_dim=128, shape=(10,100)):
    '''embedding_lookup is a generalized version of tf.gather'''
    np1 = np.random.rand(max_int,embedding_dim)
    np2 = np.random.randint(0, max_int, size=shape)
    np3 = np1[np2]

    tf1 = tf.constant(np1)
    tf2 = tf.constant(np2)
    tf3 = tf.nn.embedding_lookup(tf1, tf2)
    with tf.Session() as sess:
        tf3_ = sess.run(tf3)
    print('embedding_lookup:: tf vs np: ', hfe_r5(np3, tf3_))


def tf_embedding_lookup_mod(N0=(5,3,7), N1=9, N2=11):
    def _maximum_index(len_):
        tmp1 = list(enumerate(len_))
        x = min(tmp1, key=lambda x:x[1])
        return x[1]*len(tmp1) + x[0] #exclude self
    np_i = [np.random.uniform(size=[x,N1]) for x in N0]
    max_len = _maximum_index(N0)
    np1 = np.random.randint(0, max_len, size=[N2])
    tmp1 = np.stack([y for x in zip_longest(*np_i) for y in x if y is not None], axis=0)
    np2 = tmp1[np1]

    with tf.Graph().as_default() as tfG:
        tf_i = [tf.constant(x) for x in np_i]
        tf1 = tf.constant(np1)
        tf2 = tf.nn.embedding_lookup(tf_i, tf1, partition_strategy='mod')
    with tf.Session(graph=tfG) as sess:
        tf2_ = sess.run(tf2)
    print('embedding_lookup_mod:: np vs tf: ', hfe_r5(np2, tf2_))


def tf_embedding_lookup_div(N0=(5,3,7), N1=9, N2=11):
    def _effective_N0(N0):
        min_ = min(N0)
        N0 = [min(max(x,min_),min_+1) for x in N0]
        ret = [N0[0]]
        for x in N0[1:]:
            ret.append(min(ret[-1], x))
        return ret
    N0 = _effective_N0(N0)
    np_i = [np.random.uniform(size=[x,N1]) for x in N0]
    np1 = np.random.randint(0, sum(N0), size=[N2])

    np2 = np.concatenate(np_i, axis=0)[np1]

    with tf.Graph().as_default() as tfG:
        tf_i = [tf.constant(x) for x in np_i]
        tf1 = tf.constant(np1)
        tf2 = tf.nn.embedding_lookup(tf_i, tf1, partition_strategy='div')
    with tf.Session(graph=tfG) as sess:
        tf2_ = sess.run(tf2)
    print('embedding_lookup_div:: np vs tf: ', hfe_r5(np2, tf2_))


def tf_argsort(tf1, axis=-1):
    ndim = len(tf1.shape.as_list())
    is_transpose = (axis!=-1) and (axis!=ndim-1)
    if is_transpose:
        transpose_axis = list(range(ndim))
        transpose_axis[axis], transpose_axis[-1] = transpose_axis[-1], transpose_axis[axis]
        tf1 = tf.transpose(tf1, transpose_axis)
    ret = tf.nn.top_k(tf1, tf.shape(tf1)[-1])[1][..., ::-1]
    return tf.transpose(ret, transpose_axis) if is_transpose else ret

def _test_tf_argsort(shape=(5,6,7), axis=1):
    np1 = np.random.rand(*shape)
    np2 = np1.argsort(axis=axis)

    tf1 = tf.constant(np1)
    tf2 = tf_argsort(tf1, axis)
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('tf_argsort:: np vs tf: ', hfe_r5(np2, tf2_))


if __name__=='__main__':
    tf_count01234()
    print()
    tf_boolean_mask()
    print()
    tf_map_fn_sequence_boolean_mask()
    print()
    tf_embedding_lookup()
    print()
    tf_embedding_lookup_mod()
    print()
    tf_embedding_lookup_div()
    print()
    _test_tf_argsort()
