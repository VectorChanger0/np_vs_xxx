import numpy as np
import tensorflow as tf

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)

hf_tanh = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
hf_sigmoid = lambda x: 1/(1+np.exp(-x))


def tf_gru_single(N0=3, N2=7, N3=11):
    np1 = np.random.randn(N0, N2)
    np2 = np.random.randn(N0, N3)
    np_gkernel = np.random.randn(N2+N3, 2*N3)
    np_gbias = np.random.randn(2*N3)
    np_ckernel = np.random.randn(N2+N3, N3)
    np_cbias = np.random.randn(N3)
    tmp1 = hf_sigmoid(np.matmul(np.concatenate([np1,np2], axis=1), np_gkernel) + np_gbias)
    tmp2 = hf_tanh(np.matmul(np.concatenate([np1, np2*tmp1[:,:N3]], axis=1), np_ckernel) + np_cbias)
    np3 = tmp1[:,N3:]*np2 + (1-tmp1[:,N3:])*tmp2
    np4 = np3

    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1)
        tf2 = tf.constant(np2)
        gru0 = tf.nn.rnn_cell.GRUCell(N3, name='gru0')
        tf3,tf4 = gru0(tf1, tf2)
        z1 = {x.name:x for x in gru0.weights}
        aop = [tf.assign(z1['gru0/gates/kernel:0'], np_gkernel), tf.assign(z1['gru0/gates/bias:0'], np_gbias),
                tf.assign(z1['gru0/candidate/kernel:0'], np_ckernel), tf.assign(z1['gru0/candidate/bias:0'], np_cbias)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf3_,tf4_ = sess.run([tf3,tf4])
    print('tf_gru_single output:: np vs tf: ', hfe_r5(np3, tf3_))
    print('tf_gru_single hidden:: np vs tf: ', hfe_r5(np4, tf4_))


def tf_gru_sequence(N0=3, N1=5, N2=7, N3=11):
    np1 = np.random.randn(N0, N1, N2)
    np2 = np.random.randint(2, N1, size=[N0])
    np_h = np.random.randn(N0, N3)
    np_gkernel = np.random.randn(N2+N3, 2*N3)
    np_gbias = np.random.randn(2*N3)
    np_ckernel = np.random.randn(N2+N3, N3)
    np_cbias = np.random.randn(N3)

    np3 = np.zeros([N0,N1,N3], dtype=np.float32)
    np_h1 = np.zeros([N0,N3], dtype=np.float32)
    for ind1 in range(N0):
        hidden = np_h[ind1]
        for ind2 in range(np2[ind1]):
            tmp1 = hf_sigmoid(np.matmul(np.concatenate([np1[ind1,ind2],hidden], axis=0), np_gkernel) + np_gbias)
            tmp2 = hf_tanh(np.matmul(np.concatenate([np1[ind1,ind2], hidden*tmp1[:N3]], axis=0), np_ckernel) + np_cbias)
            hidden = tmp1[N3:]*hidden + (1-tmp1[N3:])*tmp2
            np3[ind1,ind2] = hidden
        np_h1[ind1] = hidden

    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1)
        tf2 = tf.constant(np2)
        tf_h = tf.constant(np_h)
        gru0 = tf.nn.rnn_cell.GRUCell(N3, name='gru0')
        tf3,tf_h1 = tf.nn.dynamic_rnn(gru0, tf1, initial_state=tf_h, sequence_length=tf2, scope='d_gru0')
        z1 = {x.name:x for x in gru0.weights}
        aop = [tf.assign(z1['d_gru0/gru0/gates/kernel:0'], np_gkernel), tf.assign(z1['d_gru0/gru0/gates/bias:0'], np_gbias),
                tf.assign(z1['d_gru0/gru0/candidate/kernel:0'], np_ckernel), tf.assign(z1['d_gru0/gru0/candidate/bias:0'], np_cbias)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf3_,tf_h1_ = sess.run([tf3,tf_h1])
    print('tf_gru_sequence output:: np vs tf: ', hfe_r5(np3, tf3_))
    print('tf_gru_sequence h1:: np vs tf: ', hfe_r5(np_h1, tf_h1_))


if __name__=='__main__':
    tf_gru_single()
    print('')
    tf_gru_sequence()
