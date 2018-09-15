import os
import numpy as np
import tensorflow as tf

from utils import next_tbd_dir

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def tf_write_read_tfrecords(N0=100):
    logdir = next_tbd_dir()
    hf_file = lambda *x: os.path.join(logdir, *x)
    X1 = np.random.randint(0, 3, size=(N0,))
    X2 = np.random.randint(0, 3, size=(N0,2))
    X3 = np.random.rand(N0).astype(np.float32)
    X4 = np.random.rand(N0,2).astype(np.float32)
    X5 = [str(x) for x in range(N0)]
    X6 = [(str(x),str(x+1)) for x in range(N0)]
    tfrecords_file = hf_file('test01.tfrecords')

    # write tfrecords
    with tf.python_io.TFRecordWriter(tfrecords_file) as writer:
        for ind1 in range(N0):
            example = tf.train.Example(features=tf.train.Features(feature={
                'X1': _int64_feature(X1[ind1]),
                'X2': _int64_list_feature(X2[ind1]),
                'X3': _float_feature(X3[ind1]),
                'X4': _float_list_feature(X4[ind1]),
                'X5': _bytes_feature(X5[ind1].encode()),
                'X6':_bytes_list_feature([x.encode() for x in X6[ind1]]),
            }))
            writer.write(example.SerializeToString())

    # read tfrecords
    def ds_decode_tfrecords(example_proto):
        example_fmt = {
            'X1': tf.FixedLenFeature([], tf.int64),
            'X2': tf.FixedLenFeature([2], tf.int64),
            'X3': tf.FixedLenFeature([], tf.float32),
            'X4': tf.FixedLenFeature([2], tf.float32),
            'X5': tf.FixedLenFeature([], tf.string),
            'X6': tf.FixedLenFeature([2], tf.string)
        }
        ret = tf.parse_single_example(example_proto, features=example_fmt)
        return ret['X1'],ret['X2'],ret['X3'],ret['X4'],ret['X5'],ret['X6']

    ds1 = tf.data.TFRecordDataset(tfrecords_file).map(ds_decode_tfrecords)
    tf1 = ds1.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        X1_,X2_,X3_,X4_,X5_,X6_ = zip(*[sess.run(tf1) for _ in range(N0)])

    print('X1 error: ', hfe_r5(X1, np.array(X1_)))
    print('X2 error: ', hfe_r5(X2, np.array(X2_)))
    print('X3 error: ', hfe_r5(X3, np.array(X3_)))
    print('X4 error: ', hfe_r5(X4, np.array(X4_)))
    print('X5 all equal: ', all([x==y.decode() for x,y in zip(X5,X5_)]))
    tmp1 = all([all([y1==y2.decode() for y1,y2 in zip(x1,x2)]) for x1,x2 in zip(X6, X6_)])
    print('X6 all equal: ', tmp1)


def tf_var_length_tfrecords(N0=100, min_len=3, max_len=7):
    logdir = next_tbd_dir()
    hf_file = lambda *x: os.path.join(logdir, *x)
    hf_str = lambda :''.join([chr(x) for x in np.random.randint(97,123,size=[np.random.randint(3,10)])])
    X1_len = np.random.randint(min_len,max_len,size=(N0,))
    X1 = [np.random.randint(0,100,size=[x]) for x in X1_len]
    X2_len = np.random.randint(min_len,max_len,size=(N0,))
    X2 = [np.random.rand(x) for x in X2_len]
    X3_len = np.random.randint(min_len,max_len,size=(N0,))
    X3 = [[hf_str() for _ in range(x)] for x in X3_len]
    tfrecords_file = hf_file('test01.tfrecords')

    # write
    with tf.python_io.TFRecordWriter(tfrecords_file) as writer:
        for ind1 in range(N0):
            example = tf.train.Example(features=tf.train.Features(feature={
                'X1_len': _int64_feature(X1_len[ind1]),
                'X1': _int64_list_feature(X1[ind1]),
                'X2_len': _int64_feature(X2_len[ind1]),
                'X2': _float_list_feature(X2[ind1]),
                'X3_len': _int64_feature(X3_len[ind1]),
                'X3': _bytes_list_feature([x.encode() for x in X3[ind1]]),
            }))
            writer.write(example.SerializeToString())

    # read
    def tf_decode_tfrecords(example_proto):
        example_fmt = {
            'X1_len': tf.FixedLenFeature([], tf.int64),
            'X1': tf.VarLenFeature(tf.int64),
            'X2_len': tf.FixedLenFeature([], tf.int64),
            'X2': tf.VarLenFeature(tf.float32),
            'X3_len': tf.FixedLenFeature([], tf.int64),
            'X3': tf.VarLenFeature(tf.string),
        }
        ret = tf.parse_single_example(example_proto, features=example_fmt)
        X1 = tf.sparse_to_dense(ret['X1'].indices, [ret['X1_len']], ret['X1'].values)
        X2 = tf.sparse_to_dense(ret['X2'].indices, [ret['X2_len']], ret['X2'].values)
        X3 = tf.sparse_to_dense(ret['X3'].indices, [ret['X3_len']], ret['X3'].values, '')
        return X1, X2, X3

    ds1 = tf.data.TFRecordDataset(tfrecords_file).map(tf_decode_tfrecords)
    tf1 = ds1.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        X1_,X2_,X3_ = zip(*[sess.run(tf1) for _ in range(N0)])

    tmp1 = all([hfe(x,y)<1e-5 for x,y in zip(X1,X1_)])
    print('X1 all equal: ', tmp1)
    tmp1 = all([hfe(x,y)<1e-5 for x,y in zip(X2,X2_)])
    print('X2 all equal: ', tmp1)
    tmp1 = all([all([y1==y2.decode() for y1,y2 in zip(x1,x2)]) for x1,x2 in zip(X3,X3_)])
    print('X3 all equal: ', tmp1)


if __name__=='__main__':
    tf_write_read_tfrecords()
    print()
    tf_var_length_tfrecords()

