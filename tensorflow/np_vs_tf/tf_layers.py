import numpy as np
from math import ceil
import tensorflow as tf
np.set_printoptions(precision=3)

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)


def tf_bn_training_with_control_dependencies(N0=97, N1=5, momentum=0.9, epsilon=0.01):
    '''normal train state'''
    np1 = np.random.rand(N0, N1)
    mean0 = np.random.rand(N1)
    var0 = np.random.rand(N1)

    tmp1 = np1.mean(axis=0)
    tmp2 = np1.var(axis=0)
    np2 = (np1 - tmp1) / np.sqrt(epsilon + tmp2)
    mean1 = mean0*momentum + tmp1*(1-momentum)
    var1 = var0*momentum + tmp2*(1-momentum)

    with tf.Graph().as_default() as tfG:
        tfvar0 = tf.get_variable('tfvar0', shape=np1.shape, dtype=tf.float64)
        tf1 = tf.layers.batch_normalization(tfvar0, axis=1, scale=False, center=False,
                momentum=momentum, epsilon=epsilon, training=True, name='bn1')
        tf2 = tf.reduce_mean(tf1)
        with tf.control_dependencies(tfG.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.train.GradientDescentOptimizer(0.01).minimize(tf2)
        z1 = {x.name:x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        tf_mean = z1['bn1/moving_mean:0']
        tf_var = z1['bn1/moving_variance:0']
        aop = [tf.assign(tfvar0, np1), tf.assign(tf_mean, mean0), tf.assign(tf_var, var0)]

    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        _, tf1_ = sess.run([train_op,tf1])
        tf_mean_, tf_var_ = sess.run([tf_mean, tf_var])
    print('training + control_dependencies:: bn:: np vs tf: ', hfe_r5(np2, tf1_))
    print('training + control_dependencies:: bn_mean:: np vs tf: ', hfe_r5(mean1, tf_mean_))
    print('training + control_dependencies:: bn_variance:: np vs tf: ', hfe_r5(var1, tf_var_))


def tf_bn_training_without_control_dependencies(N0=97, N1=5, momentum=0.9, epsilon=0.01):
    '''DEFINITELY WRONG'''
    np1 = np.random.rand(N0, N1)
    mean0 = np.random.rand(N1)
    var0 = np.random.rand(N1)

    tmp1 = np1.mean(axis=0)
    tmp2 = np1.var(axis=0)
    np2 = (np1 - tmp1) / np.sqrt(epsilon + tmp2)
    mean1 = mean0
    var1 = var0

    with tf.Graph().as_default() as tfG:
        tfvar0 = tf.get_variable('tfvar0', shape=np1.shape, dtype=tf.float64)
        tf1 = tf.layers.batch_normalization(tfvar0, axis=1, scale=False, center=False,
                momentum=momentum, epsilon=epsilon, training=True, name='bn1')
        tf2 = tf.reduce_mean(tf1)
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(tf2)
        z1 = {x.name:x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        tf_mean = z1['bn1/moving_mean:0']
        tf_var = z1['bn1/moving_variance:0']
        aop = [tf.assign(tfvar0, np1), tf.assign(tf_mean, mean0), tf.assign(tf_var, var0)]

    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        _, tf1_ = sess.run([train_op,tf1])
        tf_mean_, tf_var_ = sess.run([tf_mean, tf_var])
    print('training, no control_dependencies:: bn:: np vs tf: ', hfe_r5(np2, tf1_))
    print('training, no control_dependencies:: bn_mean:: np vs tf: ', hfe_r5(mean1, tf_mean_))
    print('training, no control_dependencies:: bn_variance:: np vs tf: ', hfe_r5(var1, tf_var_))


def tf_bn_not_training(N0=97, N1=5, momentum=0.9, epsilon=0.01):
    '''maybe finetune, with or without control_dependencies make no difference, UPDATE_OPS is empty'''
    np1 = np.random.rand(N0, N1)
    mean0 = np.random.rand(N1)
    var0 = np.random.rand(N1)

    np2 = (np1 - mean0) / np.sqrt(epsilon + var0)
    mean1 = mean0
    var1 = var0

    with tf.Graph().as_default() as tfG:
        tfvar0 = tf.get_variable('tfvar0', shape=np1.shape, dtype=tf.float64)
        tf1 = tf.layers.batch_normalization(tfvar0, axis=1, scale=False, center=False,
                momentum=momentum, epsilon=epsilon, training=False, name='bn1')
        tf2 = tf.reduce_mean(tf1)**2 #use **2 to maintain control dependencies
        train_op = tf.train.GradientDescentOptimizer(100).minimize(tf2)
        z1 = {x.name:x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        tf_mean = z1['bn1/moving_mean:0']
        tf_var = z1['bn1/moving_variance:0']
        aop = [tf.assign(tfvar0, np1), tf.assign(tf_mean, mean0), tf.assign(tf_var, var0)]

    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        _, tf1_ = sess.run([train_op,tf1])
        tf_mean_, tf_var_ = sess.run([tf_mean, tf_var])
    print('not training:: bn:: np vs tf: ', hfe_r5(np2, tf1_))
    print('not training:: bn_mean:: np vs tf: ', hfe_r5(mean1, tf_mean_))
    print('not training:: bn_variance:: np vs tf: ', hfe_r5(var1, tf_var_))


def tf_upsampling_with_tile(shape=(1,5,7,3), kernel=(2,3)):
    '''upsampling could be implemented with tf.tile(), tf.reshape()'''
    np1 = np.random.rand(*shape)
    new_shape = [shape[0], shape[1]*kernel[0], shape[2]*kernel[1], shape[3]]
    np2 = np.reshape(np.tile(np1, [1,1,*kernel]), new_shape)

    tf1 = tf.constant(np1, dtype=tf.float64)
    tmp1 = tf1.shape.as_list()[1:]
    tmp2 = [-1, tmp1[0]*kernel[0], tmp1[1]*kernel[1], tmp1[2]]
    tf2 = tf.reshape(tf.tile(tf1, [1,1,*kernel]), tmp2)
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('upsampling with tf.tile() :: tf vs np: ', hfe_r5(np2, tf2_))


def tf_upsampling_with_conv2d_transpose(shape=(1,5,7,3), kernel=(2,3)):
    '''upsampling could be implemented with tf.layers.conv2d_transpose()'''
    shape=(1,5,7,3)
    kernel=(2,3)
    np1 = np.random.rand(*shape)
    new_shape = [shape[0], shape[1]*kernel[0], shape[2]*kernel[1], shape[3]]
    np2 = np.reshape(np.tile(np1, [1,1,*kernel]), new_shape)

    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1)
        num1 = shape[-1]
        tf2 = tf.layers.conv2d_transpose(tf1, num1, kernel, kernel, use_bias=False, name='conv_trans1')
        tfvar1 = tfG.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[0]
        tmp1 = np.ones(kernel)[:,:,np.newaxis,np.newaxis] * np.eye(num1)
        aop = tf.assign(tfvar1, tmp1)
    with tf.Session(graph=tfG) as sess:
        sess.run(aop)
        tf2_ = sess.run(tf2)
    print('upsampling with tf.layers.conv2d_transpose:: np vs tf: ', hfe_r5(np2, tf2_))


def tf_conv_same_vs_pad_valid(n1=224, kernel=7, strides=3):
    '''padding='same' is equal to padding first than padding='valid' '''
    assert strides<=kernel
    tmp1 = int(ceil(n1/strides)-1)*strides + kernel - n1
    p1,p2 = tmp1//2, tmp1-(tmp1//2)
    np1 = np.random.rand(1,n1,n1,1)
    np2 = np.random.rand(kernel,kernel,1,1)

    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1, dtype=tf.float32)
        tf2 = tf.layers.conv2d(tf1, 1, kernel, strides, padding='same', use_bias=False, name='conv1')

        tmp1 = tf.pad(tf1, [(0,0),(p1,p2),(p1,p2),(0,0)])
        tf3 = tf.layers.conv2d(tmp1, 1, kernel, strides, padding='valid', use_bias=False, name='conv2')

        z1 = {x.name:x for x in tfG.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        tf_var_list = [z1['conv1/kernel:0'], z1['conv2/kernel:0']]
        tmp1 = [np2, np2]
        aop = [tf.assign(x,y) for x,y in zip(tf_var_list,tmp1)]
    with tf.Session(graph=tfG) as sess:
        sess.run(aop)
        tf2_, tf3_ = sess.run([tf2,tf3])
    print('same vs pad+valid:: tf vs tf: ', hfe_r5(tf2_,tf3_))


def tf_conv_channels_first_vs_last(N0=3, image_size=(28,28), channel_in=1, channel_out=3, kernel=(5,5), strides=(1,1)):
    '''channels_first or channels_last makes no difference on weights shape'''
    np1 = np.random.rand(N0,channel_in,*image_size)
    np2 = np.random.rand(*kernel, channel_in, channel_out)

    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1)
        tf2 = tf.layers.conv2d(tf1, channel_out, kernel, padding='same', use_bias=False, data_format='channels_first', name='conv1')
        tf3 = tf.transpose(tf1, (0,2,3,1))
        tf4 = tf.layers.conv2d(tf3, channel_out, kernel, padding='same', use_bias=False, data_format='channels_last', name='conv2')
        aop = [tf.assign(x,np2) for x in tfG.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf2_,tf4_ = sess.run([tf2,tf4])
    tmp1 = hfe_r5(tf2_.transpose([0,2,3,1]), tf4_)
    print('channels first vs last:: tf vs tf: ', tmp1)


def tf_full_conv_vs_dense(n1=28, c1=3, c2=5):
    '''when kernel size equal feature map size, conv is equivilent to dense'''
    np1 = np.random.rand(1,n1,n1,c1)
    np2 = np.random.rand(n1,n1,c1,c2)
    np3 = np.reshape(np2, (n1*n1*c1,c2))

    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1)
        tf2 = tf.reshape(tf.layers.conv2d(tf1, c2, n1, use_bias=False, name='conv1'), [-1,c2])
        tf3 = tf.layers.dense(tf.reshape(tf1,[-1,n1*n1*c1]), c2, use_bias=False, name='fc1')

        z1 = {x.name:x for x in tfG.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        aop = [tf.assign(z1['conv1/kernel:0'], np2), tf.assign(z1['fc1/kernel:0'], np3)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf2_,tf3_ = sess.run([tf2,tf3])
    print('full_conv vs dense:: tf vs tf: ', hfe_r5(tf2_,tf3_))


def tf_dense_constraints(step=30):
    '''demo the constraints effect'''
    with tf.Graph().as_default() as tfG:
        x = tf.random_uniform((1,4), 0, 1)
        x = tf.layers.dense(x, 1,
                kernel_initializer=tf.ones_initializer(),
                kernel_constraint=lambda y:tf.clip_by_value(y, 0, 1),
                bias_initializer=tf.ones_initializer(),
                bias_constraint=lambda y:tf.clip_by_value(y, -1, 0),
                name='dense0')[:,0]
        train_op1 = tf.train.GradientDescentOptimizer(0.1).minimize(x)
        train_op2 = tf.train.GradientDescentOptimizer(0.1).minimize(-x)
        z1 = {x.name:x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        kernel = z1['dense0/kernel:0']
        bias = z1['dense0/bias:0']

    with tf.Session(graph=tfG) as sess:
        sess.run(tf.global_variables_initializer())
        tmp1 = np.reshape(sess.run(kernel), (-1))
        tmp2 = sess.run(bias)
        print('[initialization] kernel: {}, \t bias: {}'.format(tmp1, tmp2))
        print('first minimize to see lower bound')
        for ind1 in range(step):
            _ = sess.run(train_op1)
            tmp1 = np.reshape(sess.run(kernel), (-1))
            tmp2 = sess.run(bias)
            print('[step {}] kernel: {}, \t bias: {}'.format(ind1, tmp1, tmp2))
        print('second maximize to see upper bound')
        for ind1 in range(step):
            _ = sess.run(train_op2)
            tmp1 = np.reshape(sess.run(kernel), (-1))
            tmp2 = sess.run(bias)
            print('[step {}] kernel: {}, \t bias: {}'.format(ind1, tmp1, tmp2))


def tf_constraint_override_in_layers(step=15, sc_min=0.5, sc_max=1.5, layers_min=0, layers_max=1):
    '''notice that constraints in will not effect weights in tf.layers'''
    with tf.Graph().as_default() as tfG:
        x = tf.random_uniform((1,4), 0, 1)
        with tf.variable_scope('sc1', constraint=lambda y:tf.clip_by_value(y, sc_min, sc_max)):
            x = tf.layers.dense(x, 1,
                    kernel_initializer=tf.ones_initializer(),
                    kernel_constraint=lambda y:tf.clip_by_value(y, layers_min, layers_max),
                    use_bias=False,
                    name='dense0')[:,0]
        train_op1 = tf.train.GradientDescentOptimizer(0.2).minimize(x)
        train_op2 = tf.train.GradientDescentOptimizer(0.2).minimize(-x)
        z1 = {x.name:x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        kernel = z1['sc1/dense0/kernel:0']

    with tf.Session(graph=tfG) as sess:
        sess.run(tf.global_variables_initializer())
        tmp1 = np.reshape(sess.run(kernel), (-1))
        print('[initialization] kernel: {}'.format(tmp1))
        print('first minimize to see lower bound')
        for ind1 in range(step):
            _ = sess.run(train_op1)
            tmp1 = np.reshape(sess.run(kernel), (-1))
            print('[step {}] kernel: {}'.format(ind1, tmp1))
        print('second maximize to see upper bound')
        for ind1 in range(step):
            _ = sess.run(train_op2)
            tmp1 = np.reshape(sess.run(kernel), (-1))
            print('[step {}] kernel: {}'.format(ind1, tmp1))


if __name__=='__main__':
    tf_bn_training_with_control_dependencies()
    print('')
    tf_bn_training_without_control_dependencies()
    print('')
    tf_bn_not_training()
    print('')
    tf_upsampling_with_tile()
    print('')
    tf_upsampling_with_conv2d_transpose()
    print('')
    tf_conv_same_vs_pad_valid()
    print('')
    # tf_conv_same_vs_pad_valid()# GPU required
    print('')
    tf_full_conv_vs_dense()
    print('')
    tf_dense_constraints()
    print('')
    tf_constraint_override_in_layers()
    