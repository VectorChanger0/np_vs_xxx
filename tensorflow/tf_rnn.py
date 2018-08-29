import numpy as np
import tensorflow as tf

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)

hf_tanh = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
hf_sigmoid = lambda x: 1/(1+np.exp(-x))


def tf_rnn_single(N0=3, N2=7, N3=11):
    np1 = np.random.randn(N0, N2)
    np2 = np.random.randn(N0, N3)
    np_kernel = np.random.randn(N2+N3, N3)
    np_bias = np.random.randn(N3)
    np3 = hf_tanh(np.matmul(np.concatenate([np1,np2], axis=1), np_kernel) + np_bias)
    np4 = np3

    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1)
        tf2 = tf.constant(np2)
        rnn0 = tf.nn.rnn_cell.BasicRNNCell(N3, name='rnn0')
        # tf2 = rnn0.zero_state(N0, tf.float32)
        tf3,tf4 = rnn0(tf1, tf2)
        z1 = {x.name:x for x in rnn0.weights}
        aop = [tf.assign(z1['rnn0/kernel:0'], np_kernel), tf.assign(z1['rnn0/bias:0'], np_bias)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf3_,tf4_ = sess.run([tf3,tf4])
    print('tf_rnn_single output:: np vs tf: ', hfe_r5(np3, tf3_))
    print('tf_rnn_single hidden:: np vs tf: ', hfe_r5(np4, tf4_))


def tf_rnn_sequence(N0=3, N1=5, N2=7, N3=11):
    np1 = np.random.randn(N0, N1, N2)
    np2 = np.random.randint(2, N1, size=[N0])
    np_h = np.random.randn(N0, N3)
    np_kernel = np.random.randn(N2+N3, N3)
    np_bias = np.random.randn(N3)

    np3 = np.zeros([N0,N1,N3], dtype=np.float32)
    np4 = np.zeros([N0,N3], dtype=np.float32)
    for ind1 in range(N0):
        hidden = np_h[ind1]
        for ind2 in range(np2[ind1]):
            tmp1 = np.concatenate([np1[ind1,ind2],hidden], axis=0)
            hidden = hf_tanh(np.matmul(tmp1, np_kernel) + np_bias)
            np3[ind1,ind2] = hidden
        np4[ind1] = hidden

    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1)
        tf2 = tf.constant(np2)
        tf_h = tf.constant(np_h)
        rnn0 = tf.nn.rnn_cell.BasicRNNCell(N3, name='rnn0')
        tf3, tf4 = tf.nn.dynamic_rnn(rnn0, tf1, initial_state=tf_h, sequence_length=tf2, scope='d_rnn0')
        z1 = {x.name:x for x in rnn0.weights}
        aop = [tf.assign(z1['d_rnn0/rnn0/kernel:0'], np_kernel), tf.assign(z1['d_rnn0/rnn0/bias:0'], np_bias)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf3_,tf4_ = sess.run([tf3,tf4])
    print('tf_rnn_sequence output:: np vs tf: ', hfe_r5(np3, tf3_))
    print('tf_rnn_sequence hidden:: np vs tf: ', hfe_r5(np4, tf4_))


def tf_bidirectional_rnn(N0=3, N1=5, N2=7, N3=11):
    np1 = np.random.randn(N0, N1, N2)
    np2 = np.random.randint(2, N1, size=[N0])
    np_h1 = np.random.randn(N0, N3)
    np_h2 = np.random.randn(N0, N3)
    np_kernel1 = np.random.randn(N2+N3, N3)
    np_bias1 = np.random.randn(N3)
    np_kernel2 = np.random.randn(N2+N3, N3)
    np_bias2 = np.random.randn(N3)

    np3 = np.zeros([N0,N1,N3], dtype=np.float32)
    np4 = np.zeros([N0,N1,N3], dtype=np.float32)
    np5 = np.zeros([N0,N3], dtype=np.float32)
    np6 = np.zeros([N0,N3], dtype=np.float32)
    for ind1 in range(N0):
        hidden = np_h1[ind1]
        for ind2 in range(np2[ind1]):
            tmp1 = np.concatenate([np1[ind1,ind2],hidden], axis=0)
            hidden = hf_tanh(np.matmul(tmp1, np_kernel1) + np_bias1)
            np3[ind1,ind2] = hidden
        np5[ind1] = hidden
    for ind1 in range(N0):
        hidden = np_h2[ind1]
        for ind2 in range(np2[ind1]):
            tmp1 = np.concatenate([np1[ind1,np2[ind1]-1-ind2],hidden], axis=0)
            hidden = hf_tanh(np.matmul(tmp1, np_kernel2) + np_bias2)
            np4[ind1,np2[ind1]-1-ind2] = hidden
        np6[ind1] = hidden


    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1)
        tf2 = tf.constant(np2)
        tf_h1 = tf.constant(np_h1)
        tf_h2 = tf.constant(np_h2)
        rnn0 = tf.nn.rnn_cell.BasicRNNCell(N3, name='rnn0')
        rnn1 = tf.nn.rnn_cell.BasicRNNCell(N3, name='rnn1')
        (tf3,tf4), (tf5,tf6) = tf.nn.bidirectional_dynamic_rnn(rnn0, rnn1, tf1,
                initial_state_fw=tf_h1, initial_state_bw=tf_h2, sequence_length=tf2, scope='bd')
        z1 = {x.name:x for x in rnn0.weights+rnn1.weights}
        aop = [tf.assign(z1['bd/fw/rnn0/kernel:0'], np_kernel1), tf.assign(z1['bd/fw/rnn0/bias:0'], np_bias1),
                tf.assign(z1['bd/bw/rnn1/kernel:0'], np_kernel2), tf.assign(z1['bd/bw/rnn1/bias:0'], np_bias2)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf3_,tf4_,tf5_,tf6_ = sess.run([tf3,tf4,tf5,tf6])
    print('tf_bidirectional_rnn fw:: np vs tf: ', hfe_r5(np3, tf3_))
    print('tf_bidirectional_rnn bw:: np vs tf: ', hfe_r5(np4, tf4_))
    print('tf_bidirectional_rnn fw hidden:: np vs tf: ', hfe_r5(np5, tf5_))
    print('tf_bidirectional_rnn bw hidden:: np vs tf: ', hfe_r5(np6, tf6_))


def tf_lstm_single(N0=3, N2=7, N3=11, forget_bias=0.5):
    np1 = np.random.randn(N0, N2)
    np_c = np.random.randn(N0, N3)
    np_h = np.random.randn(N0, N3)
    np_kernel = np.random.randn(N2+N3, N3*4)
    np_bias = np.random.randn(N3*4)
    
    tmp1 = np.matmul(np.concatenate([np1,np_h], axis=1), np_kernel) + np_bias
    tmp2 = np.split(tmp1, 4, axis=1)
    np_c1 = hf_sigmoid(tmp2[0])*hf_tanh(tmp2[1]) + hf_sigmoid(tmp2[2]+forget_bias)*np_c
    np_h1 = hf_tanh(np_c1) * hf_sigmoid(tmp2[3])
    np2 = np_h1
    
    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1)
        tf_c = tf.constant(np_c)
        tf_h = tf.constant(np_h)
        lstm0 = tf.nn.rnn_cell.BasicLSTMCell(N3, forget_bias, name='lstm0')
        tf2,(tf_c1,tf_h1) = lstm0(tf1, (tf_c,tf_h))
        z1 = {x.name:x for x in lstm0.weights}
        aop = [tf.assign(z1['lstm0/kernel:0'], np_kernel), tf.assign(z1['lstm0/bias:0'], np_bias)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf2_,tf_c1_,tf_h1_ = sess.run([tf2,tf_c1,tf_h1])
    print('tf_lstm_single output:: np vs tf: ', hfe_r5(np2, tf2_))
    print('tf_lstm_single h1:: np vs tf: ', hfe_r5(np_h1, tf_h1_))
    print('tf_lstm_single c1:: np vs tf: ', hfe_r5(np_c1, tf_c1_))


def tf_lstm_sequence(N0=3, N1=5, N2=7, N3=11, forget_bias=0.5):
    np1 = np.random.randn(N0, N1, N2)
    np2 = np.random.randint(2, N1, size=[N0])
    np_c = np.random.randn(N0, N3)
    np_h = np.random.randn(N0, N3)
    np_kernel = np.random.randn(N2+N3, N3*4)
    np_bias = np.random.randn(N3*4)
    
    np3 = np.zeros([N0,N1,N3], dtype=np.float32)
    np_h1 = np.zeros([N0,N3], dtype=np.float32)
    np_c1 = np.zeros([N0,N3], dtype=np.float32)
    for ind1 in range(N0):
        hidden,cell = np_h[ind1],np_c[ind1]
        for ind2 in range(np2[ind1]):
            tmp1 = np.concatenate([np1[ind1,ind2],hidden], axis=0)
            tmp2 = np.matmul(tmp1, np_kernel) + np_bias
            tmp3 = np.split(tmp2, 4, axis=0)
            cell = hf_sigmoid(tmp3[0])*hf_tanh(tmp3[1]) + hf_sigmoid(tmp3[2]+forget_bias)*cell
            hidden = hf_tanh(cell) * hf_sigmoid(tmp3[3])
            np3[ind1,ind2] = hidden
        np_h1[ind1]= hidden
        np_c1[ind1] = cell

    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1)
        tf2 = tf.constant(np2)
        tf_c = tf.constant(np_c)
        tf_h = tf.constant(np_h)
        lstm0 = tf.nn.rnn_cell.BasicLSTMCell(N3, forget_bias, name='lstm0')
        tmp1 = tf.nn.rnn_cell.LSTMStateTuple(tf_c, tf_h)
        tf3,(tf_c1,tf_h1) = tf.nn.dynamic_rnn(lstm0, tf1, initial_state=tmp1, sequence_length=tf2, scope='d_lstm0')
        z1 = {x.name:x for x in lstm0.weights}
        aop = [tf.assign(z1['d_lstm0/lstm0/kernel:0'], np_kernel), tf.assign(z1['d_lstm0/lstm0/bias:0'], np_bias)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf3_,tf_c1_,tf_h1_ = sess.run([tf3,tf_c1,tf_h1])
    print('tf_lstm_sequence output:: np vs tf: ', hfe_r5(np3, tf3_))
    print('tf_lstm_sequence h1:: np vs tf: ', hfe_r5(np_h1, tf_h1_))
    print('tf_lstm_sequence c1:: np vs tf: ', hfe_r5(np_c1, tf_c1_))


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
    tf_rnn_single()
    print('')
    tf_rnn_sequence()
    print('')
    tf_bidirectional_rnn()
    print('')
    tf_lstm_single()
    print('')
    tf_lstm_sequence()
    print('')
    tf_gru_single()
    print('')
    tf_gru_sequence()
