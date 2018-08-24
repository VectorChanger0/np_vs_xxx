import numpy as np
import tensorflow as tf

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)


def tf_count01234(N0=4):
    np1 = np.sort(np.random.randint(N0, size=(20,)))
    np2 = (np1==np.arange(N0)[:,np.newaxis]).sum(axis=1)

    ret_np = np.zeros((N0,))
    for ind1 in range(N0):
        ret_np[ind1] = (np1==ind1).sum()

    tf1 = tf.constant(np1, dtype=tf.int32)
    tf2 = tf.reduce_sum(tf.cast(tf.equal(tf1, tf.range(N0)[:,tf.newaxis]),tf.int32), axis=1)
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
        tmp1 = x[:y]*tf.reduce_mean(x[y:])
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


if __name__=='__main__':
    tf_count01234()
    print('')
    tf_boolean_mask()
    print('')
    tf_map_fn_sequence_boolean_mask()
    print('')
    tf_embedding_lookup()


