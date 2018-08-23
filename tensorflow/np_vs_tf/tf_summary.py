import os
import numpy as np
import tensorflow as tf
from collections import Iterable
import matplotlib.pyplot as plt

from utils import next_tbd_dir
hf_data = lambda *x: os.path.join('..','data',*x)


def tf_summary_scalar(N0=1000, N1=3):
    log_dir = next_tbd_dir()
    hf_file = lambda *x: os.path.join(log_dir, *x)

    npX = np.random.rand(N0, N1)
    npy = np.random.randint(0, 2, size=(N0,))

    with tf.Graph().as_default() as tfG:
        ds1 = tf.data.Dataset.from_tensor_slices((npX,npy)).repeat().shuffle(100).batch(32).prefetch(2)
        tfX, tfy = ds1.make_one_shot_iterator().get_next()
        
        x = tf.layers.dense(tfX, 20, kernel_regularizer=tf.nn.l2_loss, name='dense1')
        x = tf.nn.sigmoid(x)
        x = tf.layers.dense(x, 20, kernel_regularizer=tf.nn.l2_loss, name='dense2')
        x = tf.nn.sigmoid(x)
        x = tf.layers.dense(x, 1, name='dense3')[:,0]
        
        with tf.variable_scope('loss'):
            tmp1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(tfy,x.dtype), logits=x)
            CEloss = tf.reduce_mean(tmp1, name='CEloss')
            REGloss = 0.01*tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss_all = CEloss + REGloss
        tf.summary.scalar('loss/cross_entropy', CEloss)
        tf.summary.scalar('loss/regularizer', REGloss)
        tf.summary.scalar('loss/all', loss_all)

        with tf.variable_scope('accuracy'):
            tmp1 = tf.cast(tf.equal(tf.cast(x>0.5, tfy.dtype), tfy), tf.float32)
            acc = tf.reduce_mean(tmp1, name='acc')
        tf.summary.scalar('acc', acc)

        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss_all)
        merged = tf.summary.merge_all()

    with tf.Session(graph=tfG) as sess:
        writer = tf.summary.FileWriter(hf_file(), sess.graph)
        sess.run(tf.global_variables_initializer())
        for ind1 in range(1000):
            _,tmp1 = sess.run([train_op, merged])
            writer.add_summary(tmp1, global_step=ind1)
        
        run_metadata = tf.RunMetadata()
        _,tmp1 = sess.run([train_op,merged], run_metadata=run_metadata,
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))
        writer.add_run_metadata(run_metadata, 'step1000')
        writer.add_summary(tmp1, 1000)
        writer.close()
    print('run "tensorboard --logdir={}" out of python shell'.format(log_dir))


def _tf_read_events(filename):
    tmp1 = [(x.step,x.summary.value) for x in tf.train.summary_iterator(filename) if len(x.summary.value)!=0]
    tmp2 = [(step,y.tag,y.simple_value) for step,x in tmp1 for y in x]
    all_tag = set(x for _,x,_ in tmp2)
    ret = {x:list() for x in all_tag}
    for step,tag,value in tmp2:
        ret[tag].append((step,value))
    ret = {k:sorted(v,key=lambda x:x[0]) for k,v in ret.items()}
    return ret

def tf_read_events(filename):
    '''
    filename
        (str)
        (Iterable)
    (ret)dict: tag -> (step,simple_value)
        tag(str)
        step(int): sorted
        simple_value(float)
    '''
    assert isinstance(filename, (str,Iterable))
    if isinstance(filename, str):
        return _tf_read_events(filename)
    events_i = [_tf_read_events(x) for x in filename]
    all_tag = {y for x in events_i for y in x.keys()}
    for tag in all_tag:
        for x in events_i:
            if tag not in x:
                x[tag] = []
    ret = dict()
    for tag in all_tag:
        tmp1 = (y for x in events_i for y in x[tag])
        ret[tag] = sorted(tmp1, key=lambda x:x[0])
    return ret

def _test_tf_read_events():
    filename = hf_data('events.out.tfevents.1533523110')
    z1 = tf_read_events(filename)
    for k,v in z1.items():
        plt.figure()
        plt.plot([x for x,_ in v], [x for _,x in v])
        plt.title(k)
    plt.show()


if __name__=='__main__':
    tf_summary_scalar()
    _test_tf_read_events()
