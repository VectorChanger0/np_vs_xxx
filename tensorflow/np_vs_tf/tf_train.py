import numpy as np
import tensorflow as tf

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)


def tf_ExponentialMovingAverage(N0=10, decay=0.9):
    np1 = np.random.rand(N0)
    tmp1 = 0
    np2 = []
    for ind1 in range(N0):
        tmp1 = tmp1*decay + (1-decay)*np1[ind1]
        np2.append(tmp1)
    np2 = np.array(np2)

    with tf.Graph().as_default() as tfG:
        tf1 = tf.placeholder(tf.float64, shape=[])
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        ema_update_op = ema.apply([tf1])
        with tf.control_dependencies([ema_update_op]):
            tf2 = tf.identity(ema.average(tf1))
    with tf.Session(graph=tfG) as sess:
        sess.run(tf.global_variables_initializer())
        tf2_ = np.array([sess.run(tf2, feed_dict={tf1:x}) for x in np1])
    print('ExponentialMovingAverage:: np vs tf: ', hfe_r5(np2, tf2_))


def tf_global_step(N0=5):
    with tf.Graph().as_default() as tfG:
        tfvar1 = tf.get_variable('tfvar1', shape=[], dtype=tf.float32)
        global_step = tf.get_variable('global_step', [], tf.int64, trainable=False, initializer=tf.zeros_initializer())
        # global_step = tf.train.get_or_create_global_step()
        loss = tfvar1
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train_op = optimizer.minimize(loss, global_step=global_step)
    
    with tf.Session(graph=tfG) as sess:
        sess.run(tf.global_variables_initializer())
        global_step_ = [sess.run(global_step)]
        for _ in range(N0):
            _ = sess.run(train_op)
            global_step_.append(sess.run(global_step))
    print(global_step_)


def tf_global_step_with_two_train_op(op1_with_global_step=True, op2_with_global_step=False):
    with tf.Graph().as_default() as tfG:
        tfvar1 = tf.get_variable('tfvar1', shape=[], dtype=tf.float32)
        global_step = tf.train.get_or_create_global_step()
        tmp1 = global_step if op1_with_global_step else None
        train_op1 = tf.train.GradientDescentOptimizer(0.1).minimize(tfvar1, global_step=tmp1)
        tmp1 = global_step if op2_with_global_step else None
        tmp2 = 0 - tfvar1
        train_op2 = tf.train.GradientDescentOptimizer(0.1).minimize(tmp2, global_step=tmp1)
        train_op = tf.group(train_op1, train_op2)

    with tf.Session(graph=tfG) as sess:
        sess.run(tf.global_variables_initializer())
        print('[initialization]', sess.run(global_step))
        _ = sess.run(train_op1)
        print('[run train_op1]', sess.run(global_step))
        _ = sess.run(train_op1)
        print('[run train_op1]', sess.run(global_step))

        _ = sess.run(train_op2)
        print('[run train_op2]', sess.run(global_step))
        _ = sess.run(train_op2)
        print('[run train_op2]', sess.run(global_step))

        _ = sess.run(train_op)
        print('[run train_op]', sess.run(global_step))
        _ = sess.run(train_op)
        print('[run train_op]', sess.run(global_step))


if __name__=='__main__':
    tf_ExponentialMovingAverage()
    print('')
    tf_global_step()
    print('')
    tf_global_step_with_two_train_op()
