import random
import numpy as np
import tensorflow as tf

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)
print('[WARNING] tf.Print() will not show data in jupyter notebook')


def tf_print():
    np1 = np.random.rand(3)
    tf1 = tf.constant(np1)
    tf2 = tf.Print(tf1, [tf1+1])
    with tf.Session() as sess:
        print('you will see ```tf.print(tf2+1)``` first')
        tf2_ = sess.run(tf2)
        print('than I will call print(tf2_)')
        print(tf2_)


def tf_print_info_at_every_steps(N0=10, N1=100):
    with tf.Graph().as_default() as tfG:
        tfvar1 = tf.get_variable('tfvar1', shape=[], dtype=tf.float32)
        global_step = tf.train.get_or_create_global_step()
        tmp1 = tf.cond(tf.equal(tf.floormod(global_step, N0), 0),
            lambda: tf.Print(tfvar1, ['step: ', global_step, ' tfvar1: ', tfvar1]),
            lambda: tfvar1)
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(tmp1**2, global_step)
    with tf.Session(graph=tfG) as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(N1):
            _ = sess.run([train_op])


def tf_control_dependencies():
    with tf.Graph().as_default() as tfG:
        tf1 = tf.placeholder(tf.float32, [])
        tfvar1 = tf.get_variable('tfvar1', [], tf.float32)
        aop1 = tf.assign(tfvar1, tf.Print(tf1, ['before aop1: ', tfvar1]))
        with tf.control_dependencies([aop1]):
            aop2 = tf.assign(tfvar1, tf.Print(tf1+1, ['before aop2: ', tfvar1]))
    
    np1 = np.random.rand(1)[0]
    with tf.Session(graph=tfG) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(aop1, feed_dict={tf1:np1})
        tfvar1_ = sess.run(tfvar1)
        print('after aop1: ', tfvar1_)
        print('aop1:: np vs tf: ', hfe_r5(np1, tfvar1_))
        sess.run(tf.global_variables_initializer())
        sess.run([aop1,aop2], feed_dict={tf1:np1})
        tfvar1_ = sess.run(tfvar1)
        print('after aop2: ', tfvar1_)
        print('aop2:: np vs tf: ', hfe_r5(np1+1, tfvar1_))


def tf_cond():
    np1 = np.random.uniform(size=[])
    np2 = np1**2 if np1>0.5 else  np1**3

    tf1 = tf.constant(np1)
    tf2 = tf.cond(tf1>0.5, lambda: tf1**2, lambda: tf1**3)
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('tf.cond:: np vs tf: ', hfe_r5(np2,tf2_))


def tf_random_shuffle(N0=5, N1=7):
    np1 = np.arange(N0)[:,np.newaxis]
    np2 = np.random.rand(N0, N1)
    np3 = np.concatenate([np1,np2], axis=1)

    tf1 = tf.constant(np3, dtype=tf.float32)
    tf2 = tf.random_shuffle(tf1)
    print('[WARNING] no gradient defined for operation tf.random_shuffle()')
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    np4 = np3[tf2_[:,0].astype(np.int64)]
    print('tf.random_shuffle:: np vs tf: ', hfe_r5(np4, tf2_))


def tf_random_shuffle01(N0=5, N1=7):
    np1 = np.random.rand(N0,N1)

    tf1 = tf.constant(np1, dtype=tf.float64)
    tf2 = tf.random_shuffle(tf.range(tf.shape(tf1)[0]))
    tf3 = tf.gather(tf1, tf2, axis=0)
    with tf.Session() as sess:
        tf2_,tf3_ = sess.run([tf2,tf3])
    np2 = np1[tf2_]
    print('tf.random_shuffle:: np vs tf: ', hfe_r5(np2, tf3_))


def tf_random_seed(seed=233, N0=100):
    with tf.Graph().as_default() as tfG1:
        tf.set_random_seed(seed)
        tf1 = tf.random_uniform([N0])
    with tf.Session(graph=tfG1) as sess:
        tf1_ = sess.run(tf1)

    with tf.Graph().as_default() as tfG2:
        tf.set_random_seed(seed)
        tf2 = tf.random_uniform([N0])
    with tf.Session(graph=tfG2) as sess:
        tf2_ = sess.run(tf2)
    print('tf_random_seed:: tf vs tf: ', hfe_r5(tf1_, tf2_))


def tf_strange_dependencies(np1=1, lr=100):
    with tf.Graph().as_default() as tfG:
        tfvar0 = tf.get_variable('tfvar0', shape=[], dtype=tf.float64)
        tf1 = tf.identity(tfvar0)
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(tf1)
        aop = tf.assign(tfvar0, np1)
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        _,tf1_ = sess.run([train_op,tf1])
        print('train_op not depend on loss:: np vs tf: ', hfe_r5(tf1_, np1-lr))


def tf_while_loop_simple(min_=10, max_=100, max_record=100):
    '''Collatz conjecture'''
    x0 = np.random.randint(min_, max_)
    ind1 = 0
    np1 = x0
    np_list = []
    while np1!=1:
        ind1 += 1
        np1 = np1//2 if np1%2==0 else np1*3+1
        np_list.append(np1)
    np2 = np.array(np_list+[0]*max_record)[:max_record]

    tf1 = tf.constant(0)
    tf2 = tf.constant(x0)
    tf3 = tf.constant(0, shape=[max_record])
    cond = lambda x,y,z: tf.logical_and(tf.logical_not(tf.equal(y,1)), x<100)
    def body(x,y,z):
        y = tf.cond(tf.equal(tf.mod(y,2), 0), lambda: tf.floordiv(y,2), lambda: 3*y+1)
        z = tf.cond(x<max_record, lambda: tf.scatter_nd(x[tf.newaxis,tf.newaxis], y[tf.newaxis], [max_record])+z, lambda: z)
        return x+1,y,z
    tf4, _, tf5 = tf.while_loop(cond, body, [tf1,tf2,tf3])
    with tf.Session() as sess:
        tf4_,tf5_ = sess.run([tf4,tf5])
    print('tf_while_loop_simple_0:: py vs tf: ', hfe_r5(ind1, tf4_))
    print('tf_while_loop_simple_1:: py vs tf: ', hfe_r5(np2, tf5_))


def tf_while_loop(min_=10, max_=100, N0=5):
    '''Collatz conjecture as for loop'''
    x0 = np.random.randint(min_, max_)
    np1 = x0
    np2 = np.random.randn(N0,N0)
    np3 = np2
    while np1!=1:
        np1 = np1//2 if np1%2==0 else np1*3+1
        np3 = np.maximum(np.matmul(np3,np2), 0)

    tf1 = tf.constant(x0)
    tf2 = tf.constant(np2)
    tf3 = tf2
    cond = lambda x,y: tf.logical_not(tf.equal(x,1))
    def body(x,y):
        tmp1 = tf.cond(tf.equal(tf.mod(x,2), 0), lambda: tf.floordiv(x,2), lambda: 3*x+1)
        tmp2 = tf.nn.relu(tf.matmul(y, tf2))
        return tmp1, tmp2
    _, tf4 = tf.while_loop(cond, body, [tf1,tf3])
    with tf.Session() as sess:
        tf4_ = sess.run(tf4)
    print('tf_while_loop:: py vs tf: ', hfe_r5(np3, tf4_))


if __name__=='__main__':
    tf_print()
    print('')
    tf_print_info_at_every_steps()
    print('')
    tf_control_dependencies()
    print('')
    tf_cond()
    print('')
    tf_random_shuffle()
    print('')
    tf_random_shuffle01()
    print('')
    tf_random_seed()
    print('')
    tf_strange_dependencies()
    print('')
    tf_while_loop_simple()
    print('')
    tf_while_loop()
