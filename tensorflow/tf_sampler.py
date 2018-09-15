import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)


def tf_log_uniform_candidate_sampler(N0=3, N1=53, N2=3, N3=50000):
    '''N0(batch), N1(num_class), N2(num_true), N3(num_sample)'''
    np1 = np.random.choice(N1, size=[N0,N2], replace=False).astype(np.int64)
    tmp1 = np.arange(N1)
    np_pdf = (np.log(tmp1+2)-np.log(tmp1+1))/np.log(N1+1)

    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1)
        tf2, tf3, tf4 = tf.nn.log_uniform_candidate_sampler(true_classes=tf1, num_true=N2, num_sampled=N3, unique=False, range_max=N1)
        # cannot understand what's happening when unique is False
    with tf.Session(graph=tfG) as sess:
        tf2_,tf3_,tf4_ = sess.run([tf2,tf3,tf4])
    print('tf_log_uniform_candidate_sampler true count:: np vs tf', hfe_r5(np_pdf[np1]*N3, tf3_))
    print('tf_log_uniform_candidate_sampler sample count:: np vs tf', hfe_r5(np_pdf[tf2_]*N3, tf4_))
    tmp1 = np.zeros((N1,), dtype=np.float32)
    tmp2,tmp3 = np.unique(tf2_, return_counts=True)
    tmp1[tmp2] = tmp3/N3
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(np.arange(N1), np_pdf, label='pdf')
    ax.plot(np.arange(N1), tmp1, 'x', label='frequency')
    ax.legend()
    ax.grid()
    ax.set_title('tf_log_uniform_candidate_sampler')
    fig.show()


def _FAIL_tf_nce_loss():
    N0 = 3 #batch
    N1 = 300 #num_class
    N2 = 20 #num_sampled
    N3 = 10 #ebd_dim

    np_x = np.random.rand(N0, N3)
    # np_label = np.random.randint(0, N1, [N0,1])
    np_label = np.array([0,1,2], dtype=np.int32)[:,np.newaxis]
    np_kernel = np.random.rand(N1, N3)
    np_bias = np.random.rand(N1)
    np_pdf = (np.log(np.arange(N1)+2) - np.log(np.arange(N1)+1))/np.log(N1+1)

    with tf.Graph().as_default() as tfG:
        tf_x = tf.constant(np_x, name='tf_x', dtype=tf.float32)
        tf_label = tf.constant(np_label, name='tf_label')
        tf_kernel = tf.constant(np_kernel, name='tf_kernel', dtype=tf.float32)
        tf_bias = tf.constant(np_bias, name='tf_bias', dtype=tf.float32)
        tf1 = tf.nn.nce_loss(tf_kernel, tf_bias, tf_label, tf_x, N2, N1)
        tf_sample = tfG.get_tensor_by_name('nce_loss/LogUniformCandidateSampler:0') #cannot reproduce
        tf2 = tfG.get_tensor_by_name('nce_loss/add:0')
        tf3 = tfG.get_tensor_by_name('nce_loss/StopGradient_1:0')
        tf4 = tfG.get_tensor_by_name('nce_loss/Log:0')

    # tf.summary.FileWriter('tbd01', tfG).close()
    with tf.Session(graph=tfG) as sess:
        x1 = sess.run(tf2)
    x2 = (np_x*np_kernel[np_label[:,0]]).sum(axis=1) + np_bias[np_label[:,0]]
    print(hfe(x1, x2[:,np.newaxis]))


if __name__=='__main__':
    tf_log_uniform_candidate_sampler()
