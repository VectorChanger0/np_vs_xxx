import os
import numpy as np
import tensorflow as tf

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)

def ds_undersampling(PNRatio=7, P_keep_ratio=0.1428, N_keep_ratio=1):
    assert P_keep_ratio<=1
    assert N_keep_ratio<=1
    np1 = (np.random.rand(1000)>(1/PNRatio)).astype(np.int32)
    tmp1 = tf.constant([N_keep_ratio, P_keep_ratio], dtype=tf.float32)
    ds1 = tf.data.Dataset.from_tensor_slices(np1).repeat().shuffle(1000)
    ds2 = tf.data.Dataset.from_tensor_slices(np1).repeat().shuffle(1000) \
            .filter(lambda x: tf.random_uniform([], dtype=tf.float32)<tmp1[x])
    tf1 = ds1.make_one_shot_iterator().get_next()
    tf2 = ds2.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        tf1_ = np.array([sess.run(tf1) for _ in range(10000)], dtype=np.int32)
        tf2_ = np.array([sess.run(tf2) for _ in range(10000)], dtype=np.int32)

    tmp1 = (tf1_==1).mean()
    tmp2 = (tf1_==0).mean()
    print('{:20s}{:.3f}/{:.3f}'.format('origin P/N ratio:', tmp1, tmp2))
    tmp1 = (tf2_==1).mean()
    tmp2 = (tf2_==0).mean()
    print('{:20s}{:.3f}/{:.3f}'.format('sampling P/N ratio:', tmp1, tmp2))


def ds_oversampling(PNRatio=7, P_sampling_ratio=1, N_sampling_ratio=7):
    assert N_sampling_ratio>=1
    assert P_sampling_ratio>=1
    np1 = (np.random.rand(1000)>(1/PNRatio)).astype(np.int32)

    ds1 = tf.data.Dataset.from_tensor_slices(np1).repeat().shuffle(1000)
    PN_sampling_ratio = tf.constant([N_sampling_ratio,P_sampling_ratio], dtype=tf.float32)
    def upsampling(x):
        # x(tf,int32,(,))
        # (ret)(tf,int64,(,))
        tmp1 = PN_sampling_ratio[x]
        tmp2 = tf.floor(tmp1)
        tmp3 = tf.cast(tf.random_uniform([]) < (tmp1-tmp2), tf.int64)
        return tf.cast(tmp2, tf.int64) + tmp3
    ds2 = tf.data.Dataset.from_tensor_slices(np1).repeat().shuffle(1000) \
            .flat_map(lambda x: tf.data.Dataset.from_tensors(x).repeat(upsampling(x)))
    tf1 = ds1.make_one_shot_iterator().get_next()
    tf2 = ds2.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        tf1_ = np.array([sess.run(tf1) for _ in range(10000)], dtype=np.int32)
        tf2_ = np.array([sess.run(tf2) for _ in range(10000)], dtype=np.int32)

    tmp1 = (tf1_==1).mean()
    tmp2 = (tf1_==0).mean()
    print('{:20s}{:.3f}/{:.3f}'.format('origin P/N ratio:', tmp1, tmp2))
    tmp1 = (tf2_==1).mean()
    tmp2 = (tf2_==0).mean()
    print('{:20s}{:.3f}/{:.3f}'.format('sampling P/N ratio:', tmp1, tmp2))


def ds_unbatch(n_batch=3, batch_size=4, N2=2):
    np1 = np.arange(n_batch*batch_size*N2).reshape((n_batch,batch_size,N2))
    ds1 = tf.data.Dataset.from_tensor_slices(np1).flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    tf1 = ds1.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        for ind1 in range(n_batch*batch_size):
            print('batch {:3d}: {}'.format(ind1, sess.run(tf1)))


def ds_interleave():
    ds1 = tf.data.Dataset.range(6) \
        .interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(6), cycle_length=2, block_length=5)
    tf1 = ds1.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        tf1_ = np.array([sess.run(tf1) for _ in range(36)])
    print('interleave: ', tf1_)


def ds_concatenate():
    np1 = np.arange(5)
    np2 = np.arange(10,20).reshape((-1,2))
    ds1 = tf.data.Dataset.from_tensor_slices(np1)
    ds2 = tf.data.Dataset.from_tensor_slices(np2)
    tf1 = ds1.concatenate(ds2).make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        for _ in range(10):
            print(sess.run(tf1))


def ds_zip(N0=100):
    np1 = np.random.rand(N0)
    np2 = np.random.rand(N0)
    np3 = np.stack([np1,np2], axis=1)

    ds1 = tf.data.Dataset.from_tensor_slices(np1)
    ds2 = tf.data.Dataset.from_tensor_slices(np2)
    ds3 = tf.data.Dataset.zip((ds1,ds2)).map(lambda x1,x2:tf.stack([x1,x2]))
    tf1 = ds3.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        tf1_ = np.array([sess.run(tf1) for _ in range(N0)])
    print('zip:: relative error: ', hfe_r5(np3,tf1_))


def ds_from_string_handle():
    ds1 = tf.data.Dataset.range(100)
    ds2 = tf.data.Dataset.range(100,200)

    iter_handle = tf.placeholder(tf.string, shape=[])
    tf1 = tf.data.Iterator.from_string_handle(iter_handle, ds1.output_types, ds1.output_shapes).get_next()
    with tf.Session() as sess:
        ds1_iter_handle = sess.run(ds1.make_one_shot_iterator().string_handle())
        ds2_iter_handle = sess.run(ds2.make_one_shot_iterator().string_handle())

        z1 = []
        for _ in range(3): z1.append(sess.run(tf1, feed_dict={iter_handle:ds1_iter_handle}))
        for _ in range(3): z1.append(sess.run(tf1, feed_dict={iter_handle:ds2_iter_handle}))
        for _ in range(3): z1.append(sess.run(tf1, feed_dict={iter_handle:ds1_iter_handle}))
        for _ in range(3): z1.append(sess.run(tf1, feed_dict={iter_handle:ds2_iter_handle}))
    print(z1)


def ds_from_structure():
    ds1 = tf.data.Dataset.range(100)
    ds2 = tf.data.Dataset.range(100,200)

    ds_iter = tf.data.Iterator.from_structure(ds1.output_types, ds1.output_shapes)
    ds_iter_initializer1 = ds_iter.make_initializer(ds1)
    ds_iter_initializer2 = ds_iter.make_initializer(ds2)

    tf1 = ds_iter.get_next()

    z1 = []
    with tf.Session() as sess:
        sess.run(ds_iter_initializer1)
        for _ in range(3): z1.append(sess.run(tf1))
        sess.run(ds_iter_initializer2)
        for _ in range(3): z1.append(sess.run(tf1))
        sess.run(ds_iter_initializer1)
        for _ in range(3): z1.append(sess.run(tf1))
        sess.run(ds_iter_initializer2)
        for _ in range(3): z1.append(sess.run(tf1))
    print(z1)


def ds_batch_shuffle_or_shuffle_batch(range_max=2, batch=2):
    ds1 = tf.data.Dataset.range(range_max).repeat().batch(batch).shuffle(10)
    ds2 = tf.data.Dataset.range(range_max).repeat().shuffle(10).batch(batch)

    tf1 = ds1.make_one_shot_iterator().get_next()
    tf2 = ds2.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        tf1_ = np.array([sess.run(tf1) for _ in range(6)])
        tf2_ = np.array([sess.run(tf2) for _ in range(6)])

    print('### batch().shuffle() ###')
    print(tf1_)
    print('### shuffle().batch() ###')
    print(tf2_)


def ds_padded_batch(N0=5, batch=3):
    ds1 = tf.data.Dataset.range(10).repeat().shuffle(100).map(lambda x: tf.range(x+1)).padded_batch(batch, (None,))
    tf1 = ds1.make_one_shot_iterator().get_next()
    print('ds_padded_batch::')
    with tf.Session() as sess:
        for _ in range(N0):
            print(sess.run(tf1))


def ds_repeat_skip_or_skip_repeat():
    ds1 = tf.data.Dataset.range(3)
    tf1 = ds1.skip(1).repeat().make_one_shot_iterator().get_next()
    tf2 = ds1.repeat().skip(1).make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        z1 = [sess.run(tf1) for _ in range(10)]
        z2 = [sess.run(tf2) for _ in range(10)]
    print('.skip().repeat(): ', z1)
    print('.repeat().skip(): ', z2)


if __name__=='__main__':
    ds_undersampling()
    print('')
    ds_oversampling()
    print('')
    ds_unbatch()
    print('')
    ds_interleave()
    print('')
    ds_zip()
    print('')
    ds_from_string_handle()
    print('')
    ds_from_structure()
    print('')
    ds_batch_shuffle_or_shuffle_batch()
    print('')
    ds_padded_batch()
    print('')
    ds_repeat_skip_or_skip_repeat()
    print('')
    ds_concatenate()
