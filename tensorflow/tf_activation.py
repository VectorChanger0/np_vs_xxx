import numpy as np
import tensorflow as tf

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)


def tf_relu(shape=(3,5)):
    np1 = np.random.normal(size=shape)
    np2 = np.maximum(np1, 0)

    tf1 = tf.constant(np1)
    tf2 = tf.nn.relu(tf1)
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('tf.nn.relu:: np vs tf: ', hfe_r5(np2, tf2_))


def tf_leaky_relu(shape=(3,5), alpha=0.2):
    np1 = np.random.normal(size=shape)
    np2 = np.maximum(np1, alpha*np1)

    tf1 = tf.constant(np1)
    tf2 = tf.nn.leaky_relu(tf1, alpha=alpha)
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('tf.nn.leaky_relu:: np vs tf: ', hfe_r5(np2, tf2_))


def tf_sigmoid(shape=(3,5)):
    np1 = np.random.rand(*shape)
    np2 = 1/(1+np.exp(-np1))

    tf1 = tf.constant(np1, dtype=tf.float64)
    tf2 = tf.sigmoid(tf1)
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('sigmoid:: tf vs np: ', hfe_r5(np2, tf2_))


def tf_softmax(shape=(3,4,5), axis=-1):
    np1 = np.random.rand(*shape)
    tmp1 = np1 - np1.max(axis=axis, keepdims=True)
    tmp2 = np.exp(tmp1)
    np2 = tmp2/tmp2.sum(axis=axis, keepdims=True)

    tf1 = tf.constant(np1, dtype=tf.float64)
    tf2 = tf.nn.softmax(tf1, axis=axis)
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('softmax:: tf vs np: ', hfe_r5(np2, tf2_))

def tf_keras_hard_sigmoid(size=(3,5), low=-5, high=5):
    np1 = np.random.uniform(low, high, size)
    np2 = np.clip(0.2*np1 + 0.5, 0, 1)

    tf1 = tf.constant(np1)
    tf2 = tf.keras.activations.hard_sigmoid(tf1)
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('tf.keras.hard_sigmoid:: np vs tf: ', hfe_r5(np2, tf2_))


if __name__=='__main__':
    tf_relu()
    print('')
    tf_leaky_relu()
    print('')
    tf_sigmoid()
    print('')
    tf_softmax()
    print('')
    tf_keras_hard_sigmoid()
