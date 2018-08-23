import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from utils import next_tbd_dir

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)


def tf_metrics_acc(N1=1000):
    np1 = np.random.randint(0, 2, size=(N1,))
    tmp1 = np1/5 - 0.1 + np.random.rand(N1)
    np2 = (tmp1>((tmp1.min()+tmp1.max())/2)).astype(np.int32)

    tmp1 = (np1==np2).astype(np.int32)
    acc = np.cumsum(tmp1)/np.arange(1,N1+1)

    tf1 = tf.constant(np1, dtype=tf.int32)
    tf2 = tf.constant(np2, dtype=tf.int32)
    ds1 = tf.data.Dataset.from_tensor_slices((tf1, tf2))

    tf3,tf4 = ds1.make_one_shot_iterator().get_next()
    _,acc_update = tf.metrics.accuracy(tf3,tf4)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        acc_ = np.array([sess.run(acc_update) for _ in range(N1)])
    print('metrics acc:: np vs tf: ', hfe_r5(acc, acc_))


def tf_metrics_auc(N1=1000):
    num1 = round(N1*0.5)
    np1 = np.random.randint(0, 2, size=(N1,))
    tmp1 = np1/5 - 0.1 + np.random.rand(N1)
    np2 = (tmp1-tmp1.min())/(tmp1.max()-tmp1.min())
    auc = np.array([roc_auc_score(np1[:ind1],np2[:ind1]) for ind1 in range(num1,N1)])

    tf1 = tf.constant(np1, dtype=tf.int32)
    tf2 = tf.constant(np2, dtype=tf.float32)
    ds1 = tf.data.Dataset.from_tensor_slices((tf1, tf2))

    tf3,tf4 = ds1.make_one_shot_iterator().get_next()
    _,auc_update = tf.metrics.auc(tf3, tf4, num_thresholds=200)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        for _ in range(num1): sess.run(auc_update)
        auc_ = np.array([sess.run(auc_update) for _ in range(num1,N1)])
    print('metrics auc:: np vs tf: ', hfe_r5(auc, auc_))


def tf_estimator_auc_one_labels(N0=1000):
    def _estimator_model_fn(features, labels, mode, params=None):
        tfvar1 = tf.get_variable('tfvar1', shape=[], dtype=tf.float32)
        loss = tf.reduce_sum(features + tfvar1)*0

        if mode==tf.estimator.ModeKeys.EVAL:
            auc = tf.metrics.auc(labels, features)
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'auc':auc})

        if mode==tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(1e-10)
            train_op = optimizer.minimize(loss, tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    np1 = np.random.randint(0, 2, size=(N0,), dtype=np.int32)
    tmp1 = np1/2 - 0.25 + np.random.rand(N0)
    np2 = ((tmp1-tmp1.min())/(tmp1.max()-tmp1.min())).astype(np.float32)
    auc = roc_auc_score(np1, np2)

    model_dir = next_tbd_dir()
    ds1 = lambda: tf.data.Dataset.from_tensor_slices((np2,np1)).batch(2)
    train_config = tf.estimator.RunConfig(save_checkpoints_secs=20*60, keep_checkpoint_max=10)
    DNN = tf.estimator.Estimator(_estimator_model_fn, model_dir, train_config)
    DNN.train(ds1, steps=1)
    tmp1 = DNN.evaluate(ds1)
    print('auc1:: np vs tf.estimator: ', hfe_r5(auc, tmp1['auc']))


def tf_estimator_auc_multi_labels(N0=1000, N1=10):
    def _estimator_model_fn(features, labels, mode, params=None):
        tfvar1 = tf.get_variable('tfvar1', shape=[], dtype=tf.float32)
        loss = tf.reduce_sum(features + tfvar1)*0

        if mode==tf.estimator.ModeKeys.EVAL:
            tmp1 = [x[:,0] for x in tf.split(labels, N1, axis=1)]
            tmp2 = [x[:,0] for x in tf.split(features, N1, axis=1)]
            auc_i = {'auc_'+str(ind1):tf.metrics.auc(x, y) for ind1,(x,y) in enumerate(zip(tmp1, tmp2))}
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=auc_i)

        if mode==tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(1e-10)
            train_op = optimizer.minimize(loss, tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    np1 = np.random.randint(0, 2, size=(N0,N1), dtype=np.int32)
    tmp1 = np1/2 - 0.25 + np.random.rand(N0,N1)
    np2 = ((tmp1-tmp1.min(0))/(tmp1.max(0)-tmp1.min(0))).astype(np.float32)
    auc = np.array([roc_auc_score(np1[:,ind1], np2[:,ind1]) for ind1 in range(N1)])

    model_dir = next_tbd_dir()
    ds1 = lambda: tf.data.Dataset.from_tensor_slices((np2,np1)).batch(2)
    train_config = tf.estimator.RunConfig(save_checkpoints_secs=20*60, keep_checkpoint_max=10)
    DNN = tf.estimator.Estimator(_estimator_model_fn, model_dir, train_config)
    DNN.train(ds1, steps=1)
    tmp1 = DNN.evaluate(ds1)
    tmp2 = np.array([tmp1['auc_'+str(i)] for i in range(N1)])
    print('auc_i:: np vs tf.estimator: ', hfe_r5(auc, tmp2))


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.ERROR) #suppress info in estimator

    tf_metrics_acc()
    print('')
    tf_metrics_auc()
    print('')
    tf_estimator_auc_one_labels()
    print('')
    tf_estimator_auc_multi_labels()

    tf.logging.set_verbosity(tf.logging.INFO)
