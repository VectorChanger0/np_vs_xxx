import os
import random
import numpy as np
import tensorflow as tf

from utils import next_tbd_dir

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)
hf_data = lambda *x: os.path.join('..', 'data', *x)

def tf_save_graph(tfG, logdir):
    tf.summary.FileWriter(logdir, tfG).close()

def _test_tf_save_graph():
    logdir = next_tbd_dir()
    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(0, dtype=tf.float32, shape=[4], name='tf1')
        tfvar1 = tf.get_variable('tfvar1', shape=[3,4], dtype=tf.float32)
        _ = tf1*tfvar1
    tf_save_graph(tfG, logdir)
    print('graph has been saved to {}'.format(logdir))


def tf_load_graph_from_meta(meta_file):
    with tf.Graph().as_default() as tfG:
        _ = tf.train.import_meta_graph(meta_file)
    return tfG

def _test_tf_load_graph_from_meta():
    meta_file = hf_data('model.ckpt-2999.meta')
    logdir = next_tbd_dir()
    tfG = tf_load_graph_from_meta(meta_file)
    print('{} has been loaded'.format(meta_file))
    tf_save_graph(tfG, logdir)
    print('loaded graph has been saved to {}'.format(logdir))


def tf_load_graph_pb(filename):
    '''
    filename(str): 'frozen_graph.pb'
    (ret)(tf.Graph)
    '''
    raise Exception('this is for old GraphDef, use saved_model instead')
    with tf.Graph().as_default() as tfG:
        with open(filename, 'rb') as fid:
            tmp1 = fid.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(tmp1)
        tf.import_graph_def(graph_def, name='')
    return tfG

def tf_saved_model():
    tmp1 = next_tbd_dir()
    export_dir = os.path.join(tmp1, 'export')
    np1 = np.random.rand(3,10)
    with tf.Graph().as_default() as tfG1:
        x_in = tf.placeholder(tf.float32, shape=[None,10])
        x = tf.layers.dense(x_in, 20, name='dense0')
        x = tf.nn.relu(x)
        logits = tf.layers.dense(x, 10, name='logits')

    with tf.Session(graph=tfG1) as sess:
        sess.run(tf.global_variables_initializer())
        np2 = sess.run(logits, feed_dict={x_in:np1})
        tf.saved_model.simple_save(sess, export_dir, inputs={'x':x_in}, outputs={'logits':logits})

    # saved_model_cli show --dir ../tbd01/tbd51023/export
    with tf.Graph().as_default() as tfG2: pass
    with tf.Session(graph=tfG2) as sess:
        _ = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        tf1 = tfG2.get_tensor_by_name('Placeholder:0')
        tf2 = tfG2.get_tensor_by_name('logits/BiasAdd:0')
        np3 = sess.run(tf2, feed_dict={tf1:np1})
    print('tf_saved_model:: tf vs tf: ', hfe_r5(np2,np3))


if __name__=='__main__':
    _test_tf_save_graph()
    print('')
    _test_tf_load_graph_from_meta()

