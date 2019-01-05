import numpy as np
import tensorflow as tf

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)


def tf_conditional_random_field(N0=3,N1=5,N2=7):
    np1 = np.random.rand(N0,N1,N2)
    np2 = np.random.randint(3,N1,size=(N0,))
    np3 = np.random.rand(N2,N2)
    np4 = np.zeros(N0)
    for ind1 in range(N0):
        tmp1 = np1[ind1,0]
        for ind2 in range(1,np2[ind1]):
            tmp1 = np.matmul(tmp1, np3)*np1[ind1,ind2]
        np4[ind1] = np.log(np.sum(tmp1))

    tf1 = tf.constant(np1)
    tf2 = tf.constant(np2)
    tf3 = tf.constant(np3)
    def hf1(xy):
        x,y = xy
        cond = lambda ind1,z: ind1<y
        body = lambda ind1,z: (ind1+1,tf.matmul(z[tf.newaxis],tf3)[0]*x[ind1])
        return tf.math.log(tf.math.reduce_sum(tf.while_loop(cond, body, (tf.constant(1,tf.int64), x[0]))[1]))
    tf4 = tf.map_fn(hf1, (tf1,tf2), tf1.dtype)
    with tf.Session() as sess:
        tf4_ = sess.run(tf4)
    print('tf_crf:: np vs tf: ', hfe_r5(np4, tf4_))
