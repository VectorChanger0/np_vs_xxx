import numpy as np
import tensorflow as tf

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)


def tf_sigmoid_cross_entropy_with_logits(shape=(3,7)):
    np1 = np.random.normal(size=shape)
    np2 = np.random.randint(0, 2, size=shape)
    tmp1 = 1/(1+np.exp(-np1))
    np3 = - np2*np.log(tmp1) - (1-np2)*np.log(1-tmp1)

    tf1 = tf.constant(np1, dtype=tf.float64)
    tf2 = tf.constant(np2, dtype=tf.float64)
    tf3 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf2, logits=tf1)
    with tf.Session() as sess:
        tf3_ = sess.run(tf3)
    print('sigmoid_cross_entropy_with_logits:: np vs tf: ', hfe_r5(np3, tf3_))


def tf_weighted_cross_entropy(N0=3, N1=7):
    np1 = np.random.randint(0, 2, size=(N0,N1))
    np2 = np.random.rand(N0,N1)
    np2_ = 1/(1+np.exp(-np2))
    pos_weight = np.random.uniform(1,2,size=(N1,))
    neg_weight = np.random.uniform(1,2,size=(N1,))

    np3 = np.zeros((N0,N1))
    ind1 = np1==0
    ind2 = np.logical_not(ind1)
    np3[ind1] = -(neg_weight*np.log(1-np2_))[ind1]
    np3[ind2] = -(pos_weight*np.log(np2_))[ind2]

    tf1 = tf.constant(np1, dtype=tf.int32)
    tf2 = tf.constant(np2, dtype=tf.float32)
    tf3 = tf.nn.weighted_cross_entropy_with_logits(tf.cast(tf1, tf.float32), tf2, pos_weight/neg_weight)*neg_weight
    with tf.Session() as sess:
        tf3_ = sess.run(tf3)
    print('weighted cross entropy:: np vs tf: ',hfe_r5(np3,tf3_))


def tf_l2_loss(size=(3,5)):
    np1 = np.random.rand(*size)
    np2 = np.sum(np1**2)/2

    tf1 = tf.constant(np1)
    tf2 = tf.nn.l2_loss(tf1)
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('tf_l2_loss:: np vs tf: ', hfe_r5(np2, tf2_))


def tf_regularizer(N0=4, N1=3):
    reg = np.random.rand(1)[0]
    np1 = np.random.rand(N0,N1)
    np2 = np.random.randint(0, 2, size=(N0,))
    np3 = np.random.rand(N1,1)
    np4 = np.random.rand(1)

    tmp1 = np.matmul(np1, np3)[:,0] + np4[0]
    np5 = 1/(1+np.exp(-tmp1))
    CEloss = -(np.log(np5[np2==1]).sum() + np.log(1-np5[np2==0]).sum())/N0
    REGloss = reg*(np3**2).sum()/2

    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1, dtype=tf.float32)
        tf2 = tf.constant(np2, dtype=tf.float32)
        tmp1 = tf.layers.dense(tf1, 1, kernel_regularizer=tf.nn.l2_loss, name='dense1')[:,0]
        tmp2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=tmp1, labels=tf2)
        tf3 = tf.reduce_mean(tmp2)
        tf4 = reg*tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        
        z1 = {x.name:x for x in tfG.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        aop = [tf.assign(z1['dense1/kernel:0'], np3), tf.assign(z1['dense1/bias:0'], np4)]

    with tf.Session(graph=tfG) as sess:
        sess.run(aop)
        tf3_,tf4_ = sess.run([tf3,tf4])

    print('cross entropy loss:: np vs tf: ', hfe_r5(CEloss, tf3_))
    print('regularizer loss:: np vs tf: ', hfe_r5(REGloss, tf4_))


if __name__=='__main__':
    tf_sigmoid_cross_entropy_with_logits()
    print()
    tf_weighted_cross_entropy()
    print()
    tf_l2_loss()
    print()
    tf_regularizer()

