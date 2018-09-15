import numpy as np
import tensorflow as tf

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)


def tf_multiply_or_matmultiply():
    np1 = np.random.rand(4,4)
    np2 = np.random.rand(4,4)
    np3 = np1*np2
    np_matmul = np.matmul(np1,np2)
    np_ele_mul = np.multiply(np1, np2)

    tf1 = tf.constant(np1)
    tf2 = tf.constant(np2)
    tf3 = tf1*tf2
    tf_matmul = tf.matmul(tf1, tf2)
    tf_ele_mul = tf.multiply(tf1,tf2)

    with tf.Session() as sess:
        tf3_,tf_matmul_,tf_ele_mul_ = sess.run([tf3,tf_matmul,tf_ele_mul])

    print('element multiply, tf vs np: ',hfe_r5(np_ele_mul,tf_ele_mul_))
    print('matrix multiply, tf vs np: ',hfe_r5(np_matmul,tf_matmul_))
    tmp1 = 'matrix multiply' if hfe(np3,np_matmul)<1e-5 else 'element multiply'
    print('numpy:: np1*np2 is: ', tmp1)
    tmp1 = 'matrix multiply' if hfe(tf3_,tf_matmul_)<1e-5 else 'element multiply'
    print('tensorflow:: tf1*tf2 is: ', tmp1)


def tf_diag_diag_part(N0=5):
    np1 = np.random.rand(N0,N0)
    np2 = np.diag(np.diag(np1))

    tf1 = tf.constant(np1)
    tf2 = tf.diag(tf.diag_part(tf1))
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('tf.diag(tf.diag_part):: np vs tf: ', hfe_r5(np2, tf2_))


def tf_matrix_set_diag(N0=5):
    np1 = np.random.rand(N0,N0)
    np2 = np.random.rand(N0)
    np3 = np1 - np.diag(np.diag(np1)) + np.diag(np2)

    tf1 = tf.constant(np1)
    tf2 = tf.constant(np2)
    tf3 = tf.matrix_set_diag(tf1, tf2)
    with tf.Session() as sess:
        tf3_ = sess.run(tf3)
    print('tf.matrix_set_diag:: np vs tf: ', hfe_r5(np3, tf3_))


def tf_mod(min_=-100, max_=100, N0=100, y=7):
    np1 = np.random.randint(min_, max_, size=[N0])
    np2 = np.mod(np1, y)
    
    tf1 = tf.constant(np1, dtype=tf.int64)
    tf2 = tf.mod(tf1, y)
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('tf.mod:: np vs tf: ', hfe_r5(np2, tf2_))


def tf_tensordot(shape0=(3,5,7), shape1=(7,5,11), axes=((1,2),(1,0))):
    np1 = np.random.randn(*shape0)
    np2 = np.random.randn(*shape1)
    np3 = np.tensordot(np1, np2, axes=axes)

    tf1 = tf.constant(np1)
    tf2 = tf.constant(np2)
    tf3 = tf.tensordot(tf1, tf2, axes)
    with tf.Session() as sess:
        tf3_ = sess.run(tf3)
    print('tf_tensordot:: np vs tf: ', hfe_r5(np3,tf3_))


if __name__=='__main__':
    tf_multiply_or_matmultiply()
    print()
    tf_diag_diag_part()
    print()
    tf_matrix_set_diag()
    print()
    tf_mod()
    print()
    tf_tensordot()
