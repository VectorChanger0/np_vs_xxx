import os
import numpy as np
import tensorflow as tf

from utils import next_tbd_dir

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)


def _create_fake_ckpt(path, var_dict):
    '''
    path(str)
    var_dict(dict)
        name(str)->value(np,?,?)
    '''
    tmp1 = os.path.dirname(path)
    if not os.path.exists(tmp1): os.makedirs(tmp1)
    dtype_map = {'float32':tf.float32, 'float64':tf.float32, 'int32':tf.int64, 'int64':tf.int64}
    with tf.Graph().as_default() as tfG:
        for k,v in var_dict.items():
            _ = tf.get_variable(k, dtype=dtype_map[v.dtype.name], initializer=v)
        saver = tf.train.Saver()
    CUDA_info = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tf.Session(graph=tfG) as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, path)
    if CUDA_info is None:
        del os.environ['CUDA_VISIBLE_DEVICES']
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_info
    os.remove(os.path.join(os.path.dirname(path), 'checkpoint'))
    os.remove(path+'.meta')


def tf_events_filesize_whether_init_from_ckpt(N0=1000, N1=5000):
    logdir = next_tbd_dir()
    hf_file = lambda *x,dir0=logdir: os.path.join(dir0, *x)
    np1 = np.random.rand(N0, N1).astype(np.float32)

    # init_from_initializer
    dir1 = hf_file('init_from_initializer')
    with tf.Graph().as_default() as tfG1:
        tfG1_tfvar1 = tf.get_variable('tfG1_tfvar1', dtype=tf.float32, initializer=np1)
    with tf.Session(graph=tfG1) as sess:
        tf.summary.FileWriter(dir1, tfG1).close()
        _ = sess.run(tf.global_variables_initializer())
        tfG1_tfvar1_ = sess.run(tfG1_tfvar1)

    # init_from_ckpt
    dir2 = hf_file('init_from_ckpt')
    fake_ckpt = hf_file('fake_ckpt', 'model.ckpt')
    _create_fake_ckpt(fake_ckpt, {'tfvar1':np1})
    with tf.Graph().as_default() as tfG2:
        tfG2_tfvar1 = tf.get_variable('tfG2_tfvar1', dtype=tf.float32, shape=[N0,N1])
        tf.train.init_from_checkpoint(fake_ckpt, {'tfvar1':'tfG2_tfvar1'})
    with tf.Session(graph=tfG2) as sess:
        tf.summary.FileWriter(dir2, tfG2).close()
        _ = sess.run(tf.global_variables_initializer())
        tfG2_tfvar1_ = sess.run(tfG2_tfvar1)

    print('tf_init_from_initializer:: np vs tf: ', hfe_r5(np1, tfG1_tfvar1_))
    print('tf_init_from_ckpt:: np vs tf: ', hfe_r5(np1, tfG2_tfvar1_))
    hf1 = lambda dir0: os.path.join(dir0, [x for x in os.listdir(dir0) if x.startswith('events')][0])
    print('tf_init_from_initializer events file size: ', os.path.getsize(hf1(dir1)))
    print('tf_init_from_ckpt events file size: ', os.path.getsize(hf1(dir2)))


def tf_init_using_hook(N0=1000, N1=5000):
    logdir = next_tbd_dir()
    hf_file = lambda *x,dir0=logdir: os.path.join(dir0, *x)
    np1 = np.random.rand(N0, N1).astype(np.float32)

    def model_fn(features, labels, mode, params):
        tfvar1 = tf.get_variable('tfvar1', dtype=tf.float32, shape=[N0,N1])
        predict = tfvar1**2
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions={'predict':predict})
        loss = tf.reduce_sum(predict)
        if mode==tf.estimator.ModeKeys.EVAL:
            mae = tf.metrics.mean_absolute_error(predict, predict)
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'mae':mae})
        optimizer = tf.train.GradientDescentOptimizer(params['lr'])
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    fake_ckpt = hf_file('fake_ckpt', 'model.ckpt')
    _create_fake_ckpt(fake_ckpt, {'tfvar1':np1})
    tmp1 = np.arange(10),np.arange(10)
    ds_train = lambda: tf.data.Dataset.from_tensor_slices(tmp1)
    params = {'lr': 1}
    DNN = tf.estimator.Estimator(model_fn, logdir, config=tf.estimator.RunConfig(), params=params)
    hook1 = _MyHook1(fake_ckpt)
    DNN.train(ds_train, steps=1, hooks=[hook1])
    print('tf_init_using_hook before:: np vs tf: ', hfe_r5(np1, hook1._tfvar1_))
    tmp1 = np1 - np1*2*params['lr']
    print('tf_init_using_hook after:: np vs tf: ', hfe_r5(tmp1, hook1.tfvar1_))

class _MyHook1(tf.train.SessionRunHook):
    def __init__(self, ckpt):
        self.ckpt = ckpt
    def begin(self):
        tf.train.init_from_checkpoint(self.ckpt, {'tfvar1':'tfvar1'})
    def after_create_session(self, session, coord):
        self.tfvar1 = session.graph.get_tensor_by_name('tfvar1:0')
        self.sess = session
    def before_run(self, run_context):
        self._tfvar1_ = self.sess.run(self.tfvar1)
        return tf.train.SessionRunArgs(self.tfvar1)
    def after_run(self, run_context, run_values):
        self.tfvar1_ = run_values.results
    def end(self, session):
        pass


if __name__=='__main__':
    tf_events_filesize_whether_init_from_ckpt()
    print('')
    tf_init_using_hook()
