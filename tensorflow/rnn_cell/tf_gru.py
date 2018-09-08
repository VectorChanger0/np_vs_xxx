import numpy as np
import tensorflow as tf

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)

hf_tanh = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
hf_sigmoid = lambda x: 1/(1+np.exp(-x))


def tf_gru_single(N0=3, N2=7, N3=11):
    inputx = np.random.randn(N0, N2)
    hidden = np.random.rand(N0, N3)
    ru_kernel = np.random.randn(N2+N3, 2*N3)
    ru_bias = np.random.randn(2*N3)
    c_kernel = np.random.randn(N2+N3, N3)
    c_bias = np.random.randn(N3)
    
    tmp1 = np.concatenate([inputx,hidden], axis=1)
    r_gate, u_gate = np.split(hf_sigmoid(np.matmul(tmp1, ru_kernel) + ru_bias), 2, axis=1)
    np_c = hf_tanh(np.matmul(np.concatenate([inputx, hidden*r_gate], axis=1), c_kernel) + c_bias)
    np2 = u_gate*hidden + (1-u_gate)*np_c
    np1 = np2

    with tf.Graph().as_default() as tfG:
        tf_x = tf.constant(inputx)
        tf_h = tf.constant(hidden)
        gru0 = tf.nn.rnn_cell.GRUCell(N3, name='gru0')
        tf1,tf2 = gru0(tf_x, tf_h)
        z1 = {x.name:x for x in gru0.weights}
        aop = [tf.assign(z1['gru0/gates/kernel:0'], ru_kernel), tf.assign(z1['gru0/gates/bias:0'], ru_bias),
                tf.assign(z1['gru0/candidate/kernel:0'], c_kernel), tf.assign(z1['gru0/candidate/bias:0'], c_bias)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf1_,tf2_ = sess.run([tf1,tf2])
    print('tf_gru_single output:: np vs tf: ', hfe_r5(np1, tf1_))
    print('tf_gru_single hidden:: np vs tf: ', hfe_r5(np2, tf2_))


def tf_gru_sequence(N0=3, N1=5, N2=7, N3=11):
    inputx = np.random.randn(N0, N1, N2)
    sequence_length = np.random.randint(2, N1, size=[N0])
    hidden = np.random.randn(N0, N3)
    ru_kernel = np.random.randn(N2+N3, 2*N3)
    ru_bias = np.random.randn(2*N3)
    c_kernel = np.random.randn(N2+N3, N3)
    c_bias = np.random.randn(N3)

    np1 = np.zeros([N0,N1,N3], dtype=np.float32)
    np2 = np.zeros([N0,N3], dtype=np.float32)
    for ind1 in range(N0):
        np_h = hidden[ind1]
        for ind2 in range(sequence_length[ind1]):
            tmp1 = np.concatenate([inputx[ind1,ind2],np_h], axis=0)
            r_gate,u_gate = np.split(hf_sigmoid(np.matmul(tmp1, ru_kernel) + ru_bias), 2, axis=0)
            np_c = hf_tanh(np.matmul(np.concatenate([inputx[ind1,ind2], np_h*r_gate], axis=0), c_kernel) + c_bias)
            np_h = u_gate*np_h + (1-u_gate)*np_c
            np1[ind1,ind2] = np_h
        np2[ind1] = np_h

    with tf.Graph().as_default() as tfG:
        tf_x = tf.constant(inputx)
        tf_sequence_length = tf.constant(sequence_length)
        tf_h = tf.constant(hidden)
        gru0 = tf.nn.rnn_cell.GRUCell(N3, name='gru0')
        tf1,tf2 = tf.nn.dynamic_rnn(gru0, tf_x, initial_state=tf_h, sequence_length=tf_sequence_length, scope='d_gru0')
        z1 = {x.name:x for x in gru0.weights}
        aop = [tf.assign(z1['d_gru0/gru0/gates/kernel:0'], ru_kernel), tf.assign(z1['d_gru0/gru0/gates/bias:0'], ru_bias),
                tf.assign(z1['d_gru0/gru0/candidate/kernel:0'], c_kernel), tf.assign(z1['d_gru0/gru0/candidate/bias:0'], c_bias)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf1_,tf2_ = sess.run([tf1,tf2])
    print('tf_gru_sequence output:: np vs tf: ', hfe_r5(np1, tf1_))
    print('tf_gru_sequence hidden:: np vs tf: ', hfe_r5(np2, tf2_))


if __name__=='__main__':
    tf_gru_single()
    print('')
    tf_gru_sequence()
