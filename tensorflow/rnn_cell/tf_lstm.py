import numpy as np
import tensorflow as tf

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)

hf_tanh = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
hf_sigmoid = lambda x: 1/(1+np.exp(-x))


def tf_lstm_single(N0=3, N2=7, N3=11, forget_bias=0.5):
    inputx = np.random.randn(N0, N2)
    cell = np.random.randn(N0, N3)
    hidden = np.random.randn(N0, N3)
    ucfo_kernel = np.random.randn(N2+N3, N3*4)
    ucfo_bias = np.random.randn(N3*4)
    
    tmp1 = np.concatenate([inputx,hidden], axis=1)
    Du,Dc,Df,Do = np.split(np.matmul(tmp1, ucfo_kernel) + ucfo_bias, 4, axis=1)
    np_c1 = hf_sigmoid(Du)*hf_tanh(Dc) + hf_sigmoid(Df+forget_bias)*cell
    np_h1 = hf_tanh(np_c1) * hf_sigmoid(Do)
    np1 = np_h1
    
    with tf.Graph().as_default() as tfG:
        tf_x = tf.constant(inputx)
        tf_c = tf.constant(cell)
        tf_h = tf.constant(hidden)
        lstm0 = tf.nn.rnn_cell.BasicLSTMCell(N3, forget_bias, name='lstm0')
        tf1,(tf_c1,tf_h1) = lstm0(tf_x, (tf_c,tf_h))
        z1 = {x.name:x for x in lstm0.weights}
        aop = [tf.assign(z1['lstm0/kernel:0'], ucfo_kernel), tf.assign(z1['lstm0/bias:0'], ucfo_bias)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf1_,tf_c1_,tf_h1_ = sess.run([tf1,tf_c1,tf_h1])
    print('tf_lstm_single output:: np vs tf: ', hfe_r5(np1, tf1_))
    print('tf_lstm_single h1:: np vs tf: ', hfe_r5(np_h1, tf_h1_))
    print('tf_lstm_single c1:: np vs tf: ', hfe_r5(np_c1, tf_c1_))


def tf_lstm_sequence(N0=3, N1=5, N2=7, N3=11, forget_bias=0.5):
    inputx = np.random.randn(N0, N1, N2)
    sequence_length = np.random.randint(2, N1, size=[N0])
    cell = np.random.randn(N0, N3)
    hidden = np.random.randn(N0, N3)
    ucfo_kernel = np.random.randn(N2+N3, N3*4)
    ucfo_bias = np.random.randn(N3*4)
    
    np1 = np.zeros([N0,N1,N3], dtype=np.float32)
    np_h1 = np.zeros([N0,N3], dtype=np.float32)
    np_c1 = np.zeros([N0,N3], dtype=np.float32)
    for ind1 in range(N0):
        np_c, np_h = cell[ind1], hidden[ind1]
        for ind2 in range(sequence_length[ind1]):
            tmp1 = np.concatenate([inputx[ind1,ind2],np_h], axis=0)
            Du,Dc,Df,Do = np.split(np.matmul(tmp1, ucfo_kernel) + ucfo_bias, 4, axis=0)
            np_c = hf_sigmoid(Du)*hf_tanh(Dc) + hf_sigmoid(Df+forget_bias)*np_c
            np_h = hf_sigmoid(Do) * hf_tanh(np_c)
            np1[ind1,ind2] = np_h
        np_h1[ind1]= np_h
        np_c1[ind1] = np_c

    with tf.Graph().as_default() as tfG:
        tf_x = tf.constant(inputx)
        tf_sequence_length = tf.constant(sequence_length)
        tf_c = tf.constant(cell)
        tf_h = tf.constant(hidden)
        lstm0 = tf.nn.rnn_cell.BasicLSTMCell(N3, forget_bias, name='lstm0')
        tmp1 = tf.nn.rnn_cell.LSTMStateTuple(tf_c, tf_h)
        tf1,(tf_c1,tf_h1) = tf.nn.dynamic_rnn(lstm0, tf_x, initial_state=tmp1, sequence_length=tf_sequence_length, scope='d_lstm0')
        z1 = {x.name:x for x in lstm0.weights}
        aop = [tf.assign(z1['d_lstm0/lstm0/kernel:0'], ucfo_kernel), tf.assign(z1['d_lstm0/lstm0/bias:0'], ucfo_bias)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf1_,tf_c1_,tf_h1_ = sess.run([tf1,tf_c1,tf_h1])
    print('tf_lstm_sequence output:: np vs tf: ', hfe_r5(np1, tf1_))
    print('tf_lstm_sequence h1:: np vs tf: ', hfe_r5(np_h1, tf_h1_))
    print('tf_lstm_sequence c1:: np vs tf: ', hfe_r5(np_c1, tf_c1_))


def tf_bidirectional_lstm(N0=3, N1=5, N2=7, N3=11, forget_bias=0.5):
    inputx = np.random.randn(N0, N1, N2)
    sequence_length = np.random.randint(2, N1, size=[N0])
    cell_f = np.random.randn(N0, N3)
    hidden_f = np.random.randn(N0, N3)
    cell_b = np.random.randn(N0, N3)
    hidden_b = np.random.randn(N0, N3)
    ucfo_kernel_f = np.random.randn(N2+N3, N3*4)
    ucfo_bias_f = np.random.randn(N3*4)
    ucfo_kernel_b = np.random.randn(N2+N3, N3*4)
    ucfo_bias_b = np.random.randn(N3*4)
    
    np1 = np.zeros([N0,N1,N3], dtype=np.float32)
    np_cf1 = np.zeros([N0,N3], dtype=np.float32)
    np_hf1 = np.zeros([N0,N3], dtype=np.float32)
    for ind1 in range(N0):
        np_c,np_h = cell_f[ind1],hidden_f[ind1]
        for ind2 in range(sequence_length[ind1]):
            tmp1 = np.concatenate([inputx[ind1,ind2],np_h], axis=0)
            Du,Dc,Df,Do = np.split(np.matmul(tmp1, ucfo_kernel_f) + ucfo_bias_f, 4, axis=0)
            np_c = hf_sigmoid(Du)*hf_tanh(Dc) + hf_sigmoid(Df+forget_bias)*np_c
            np_h = hf_sigmoid(Do) * hf_tanh(np_c)
            np1[ind1,ind2] = np_h
        np_cf1[ind1] = np_c
        np_hf1[ind1]= np_h
    np2 = np.zeros([N0,N1,N3], dtype=np.float32)
    np_cb1 = np.zeros([N0,N3], dtype=np.float32)
    np_hb1 = np.zeros([N0,N3], dtype=np.float32)
    for ind1 in range(N0):
        np_c,np_h = cell_b[ind1],hidden_b[ind1]
        for ind2 in range(sequence_length[ind1]):
            tmp1 = np.concatenate([inputx[ind1,sequence_length[ind1]-1-ind2],np_h], axis=0)
            Du,Dc,Df,Do = np.split(np.matmul(tmp1, ucfo_kernel_b) + ucfo_bias_b, 4, axis=0)
            np_c = hf_sigmoid(Du)*hf_tanh(Dc) + hf_sigmoid(Df+forget_bias)*np_c
            np_h = hf_sigmoid(Do) * hf_tanh(np_c)
            np2[ind1,sequence_length[ind1]-1-ind2] = np_h
        np_cb1[ind1] = np_c
        np_hb1[ind1]= np_h

    with tf.Graph().as_default() as tfG:
        tf_x = tf.constant(inputx)
        tf_sequence_length = tf.constant(sequence_length)
        tf_cf = tf.constant(cell_f)
        tf_hf = tf.constant(hidden_f)
        tf_cb = tf.constant(cell_b)
        tf_hb = tf.constant(hidden_b)
        lstm0 = tf.nn.rnn_cell.BasicLSTMCell(N3, forget_bias, name='lstm0')
        lstm1 = tf.nn.rnn_cell.BasicLSTMCell(N3, forget_bias, name='lstm1')
        tmp1 = tf.nn.rnn_cell.LSTMStateTuple(tf_cf, tf_hf)
        tmp2 = tf.nn.rnn_cell.LSTMStateTuple(tf_cb, tf_hb)
        (tf1,tf2), ((tf_cf1,tf_hf1),(tf_cb1,tf_hb1)) = tf.nn.bidirectional_dynamic_rnn(lstm0, lstm1, tf_x,
                initial_state_fw=tmp1, initial_state_bw=tmp2, sequence_length=tf_sequence_length, scope='bd')
        z1 = {x.name:x for x in lstm0.weights+lstm1.weights}
        aop = [tf.assign(z1['bd/fw/lstm0/kernel:0'], ucfo_kernel_f), tf.assign(z1['bd/fw/lstm0/bias:0'], ucfo_bias_f),
                tf.assign(z1['bd/bw/lstm1/kernel:0'], ucfo_kernel_b), tf.assign(z1['bd/bw/lstm1/bias:0'], ucfo_bias_b)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf1_,tf2_,tf_cf1_,tf_hf1_,tf_cb1_,tf_hb1_ = sess.run([tf1,tf2,tf_cf1,tf_hf1,tf_cb1,tf_hb1])
    print('tf_bidirectional_lstm forward:: np vs tf: ', hfe_r5(np1, tf1_))
    print('tf_bidirectional_lstm backward:: np vs tf: ', hfe_r5(np2, tf2_))
    print('tf_bidirectional_lstm cf1:: np vs tf: ', hfe_r5(np_cf1, tf_cf1_))
    print('tf_bidirectional_lstm hf1:: np vs tf: ', hfe_r5(np_hf1, tf_hf1_))
    print('tf_bidirectional_lstm cb1:: np vs tf: ', hfe_r5(np_cb1, tf_cb1_))
    print('tf_bidirectional_lstm hb1:: np vs tf: ', hfe_r5(np_hb1, tf_hb1_))


if __name__=='__main__':
    tf_lstm_single()
    print('')
    tf_lstm_sequence()
    print('')
    tf_bidirectional_lstm()
