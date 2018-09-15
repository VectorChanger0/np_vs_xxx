import tensorflow as tf

def tf_tensor_op_name_in_variable_scope():
    with tf.Graph().as_default():
        with tf.variable_scope('sc1',reuse=tf.AUTO_REUSE):
            tf1 = tf.get_variable('tf1', shape=[], dtype=tf.float32)
            op1 = tf.assign(tf1,0,name='op1')
        
        with tf.variable_scope('sc1',reuse=tf.AUTO_REUSE):
            tf2 = tf.get_variable('tf2', shape=[], dtype=tf.float32)
            op2 = tf.assign(tf2,0,name='op2')
        
        tf3 = tf.get_variable('sc1/tf3', shape=[], dtype=tf.float32)
        op3 = tf.assign(tf3, 0, name='sc1/op3')
    print(tf1, tf2, tf3, op1, op2, op3, sep='\n')


def tf_variable_scope_constraints(value0=1.5, lr=0.3, step=6, clip_min=0, clip_max=1):
    with tf.Graph().as_default() as tfG:
        with tf.variable_scope('sc1', constraint=lambda x: tf.clip_by_value(x, clip_min, clip_max)):
            tfvar1 = tf.get_variable('tfvar1', shape=[])
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(tfvar1)

    with tf.Session(graph=tfG) as sess:
        sess.run(tf.assign(tfvar1, value0))
        print('assign value {} first: tfvar1: '.format(value0), sess.run(tfvar1))
        for ind1 in range(step):
            _,tmp1 = sess.run([train_op, tfvar1])
            print('after step {}: tfvar1: {}'.format(ind1, tmp1))


if __name__=='__main__':
    tf_tensor_op_name_in_variable_scope()
    print()
    tf_variable_scope_constraints()
