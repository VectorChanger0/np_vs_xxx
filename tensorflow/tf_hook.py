import os
import numpy as np
import tensorflow as tf

def _model_fn(features, labels, mode, params):
    logits = tf.layers.dense(features, params['N1'], name='logits')
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'logits':logits})

    with tf.variable_scope('loss'):
        tmp1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.math.reduce_mean(tmp1)
    acc = tf.metrics.accuracy(labels, tf.math.argmax(logits,axis=1), name='acc')
    if mode==tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'acc':acc})

    train_op = tf.train.AdamOptimizer(0.001).minimize(loss, tf.train.get_or_create_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

class MyHook1(tf.train.SessionRunHook):
    def __init__(self):
        self.state = 0
        print('init MyHook({})'.format(self.state))
    def begin(self):
        self.state += 1
        print('begin({})'.format(self.state))
    def after_create_session(self, session, coord):
        self.state += 1
        print('after_create_session({})'.format(self.state))
    def before_run(self, run_context):
        self.state += 1
        print('before_run({})'.format(self.state))
    def after_run(self, run_context, run_values):
        self.state += 1
        print('after_run({})'.format(self.state))
    def end(self, session):
        self.state += 1
        print('end({})'.format(self.state))

def _test_MyHook1():
    N0 = 100
    N1 = 10
    X = np.random.rand(N0, N1).astype(np.float32)
    y = np.random.randint(0, N1, size=(N0), dtype=np.int64)

    ds_train = lambda: tf.data.Dataset.from_tensor_slices((X,y)).repeat().shuffle(N0).batch(1).prefetch(2)
    train_config = tf.estimator.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=None)
    DNN = tf.estimator.Estimator(_model_fn, config=train_config, params={'N1':10})
    DNN.train(ds_train, steps=5, hooks=[MyHook1()])


if __name__=='__main__':
    _test_MyHook1()
