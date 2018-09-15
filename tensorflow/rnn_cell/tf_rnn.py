import numpy as np
import tensorflow as tf

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)

hf_tanh = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
hf_sigmoid = lambda x: 1/(1+np.exp(-x))


def tf_rnn_single(N0=3, N2=7, N3=11):
    inputx = np.random.randn(N0, N2)
    hidden = np.random.randn(N0, N3)
    kernel = np.random.randn(N2+N3, N3)
    bias = np.random.randn(N3)

    np1 = hf_tanh(np.matmul(np.concatenate([inputx,hidden], axis=1), kernel) + bias)
    np2 = np1

    with tf.Graph().as_default() as tfG:
        tf_x = tf.constant(inputx)
        tf_h = tf.constant(hidden)
        rnn0 = tf.nn.rnn_cell.BasicRNNCell(N3, name='rnn0')
        tf1,tf2 = rnn0(tf_x, tf_h)
        z1 = {x.name:x for x in rnn0.weights}
        aop = [tf.assign(z1['rnn0/kernel:0'], kernel), tf.assign(z1['rnn0/bias:0'], bias)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf1_,tf2_ = sess.run([tf1,tf2])
    print('tf_rnn_single output:: np vs tf: ', hfe_r5(np1, tf1_))
    print('tf_rnn_single hidden:: np vs tf: ', hfe_r5(np2, tf2_))


def tf_rnn_dropout(N0=3, N2=7, N3=11, input_keep_prob=0.7, output_keep_prob=0.8, state_keep_prob=0.9):
    inputx = np.random.randn(N0, N2)
    hidden = np.random.randn(N0, N3)
    kernel = np.random.randn(N2+N3, N3)
    bias = np.random.randn(N3)

    with tf.Graph().as_default() as tfG:
        tf_x = tf.constant(inputx)
        tf_h = tf.constant(hidden)
        rnn0 = tf.nn.rnn_cell.BasicRNNCell(N3, name='rnn0')
        dropoutcell = tf.nn.rnn_cell.DropoutWrapper(rnn0, input_keep_prob=input_keep_prob,
                    output_keep_prob=output_keep_prob, state_keep_prob=state_keep_prob)
        tf1,tf2 = dropoutcell(tf_x, tf_h)
        z1 = {x.name:x for x in rnn0.weights}
        aop = [tf.assign(z1['rnn0/kernel:0'], kernel), tf.assign(z1['rnn0/bias:0'], bias)]
        tf_in_mask = tfG.get_tensor_by_name('dropout/random_uniform:0')
        tf_state_mask = tfG.get_tensor_by_name('dropout_1/random_uniform:0')
        tf_out_mask = tfG.get_tensor_by_name('dropout_2/random_uniform:0')
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tmp1 = [tf1,tf2,tf_in_mask,tf_state_mask,tf_out_mask]
        tf1_,tf2_,tf_in_mask_,tf_state_mask_,tf_out_mask_ = sess.run(tmp1)

    def _dropout(x, kepr, mask):
        return x/kepr * np.floor(mask+kepr)
    tmp1 = _dropout(inputx, input_keep_prob, tf_in_mask_)
    tmp2 = hf_tanh(np.matmul(np.concatenate([tmp1,hidden], axis=1), kernel) + bias)
    np1 = _dropout(tmp2, output_keep_prob, tf_out_mask_)
    np2 = _dropout(tmp2, state_keep_prob, tf_state_mask_)

    print('tf_rnn_dropout output:: np vs tf: ', hfe_r5(np1, tf1_))
    print('tf_rnn_dropout hidden:: np vs tf: ', hfe_r5(np2, tf2_))


def tf_rnn_sequence(N0=3, N1=5, N2=7, N3=11):
    inputx = np.random.randn(N0, N1, N2)
    sequence_length = np.random.randint(2, N1, size=[N0])
    hidden = np.random.randn(N0, N3)
    kernel = np.random.randn(N2+N3, N3)
    bias = np.random.randn(N3)

    np1 = np.zeros([N0,N1,N3], dtype=np.float32)
    np2 = np.zeros([N0,N3], dtype=np.float32)
    for ind1 in range(N0):
        np_h = hidden[ind1]
        for ind2 in range(sequence_length[ind1]):
            tmp1 = np.concatenate([inputx[ind1,ind2],np_h], axis=0)
            np_h = hf_tanh(np.matmul(tmp1, kernel) + bias)
            np1[ind1,ind2] = np_h
        np2[ind1] = np_h

    with tf.Graph().as_default() as tfG:
        tf_x = tf.constant(inputx)
        tf_sequence_length = tf.constant(sequence_length)
        tf_h = tf.constant(hidden)
        rnn0 = tf.nn.rnn_cell.BasicRNNCell(N3, name='rnn0')
        tf1, tf2 = tf.nn.dynamic_rnn(rnn0, tf_x, initial_state=tf_h, sequence_length=tf_sequence_length, scope='d_rnn0')
        z1 = {x.name:x for x in rnn0.weights}
        aop = [tf.assign(z1['d_rnn0/rnn0/kernel:0'], kernel), tf.assign(z1['d_rnn0/rnn0/bias:0'], bias)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf1_,tf2_ = sess.run([tf1,tf2])
    print('tf_rnn_sequence output:: np vs tf: ', hfe_r5(np1, tf1_))
    print('tf_rnn_sequence hidden:: np vs tf: ', hfe_r5(np2, tf2_))


def tf_bidirectional_rnn(N0=3, N1=5, N2=7, N3=11):
    inputx = np.random.randn(N0, N1, N2)
    sequence_length = np.random.randint(2, N1, size=[N0])
    hidden_f = np.random.randn(N0, N3)
    hidden_b = np.random.randn(N0, N3)
    kernel_f = np.random.randn(N2+N3, N3)
    bias_f = np.random.randn(N3)
    kernel_b = np.random.randn(N2+N3, N3)
    bias_b = np.random.randn(N3)

    np1 = np.zeros([N0,N1,N3], dtype=np.float32)
    np3 = np.zeros([N0,N3], dtype=np.float32)
    for ind1 in range(N0):
        np_h = hidden_f[ind1]
        for ind2 in range(sequence_length[ind1]):
            tmp1 = np.concatenate([inputx[ind1,ind2],np_h], axis=0)
            np_h = hf_tanh(np.matmul(tmp1, kernel_f) + bias_f)
            np1[ind1,ind2] = np_h
        np3[ind1] = np_h
    np2 = np.zeros([N0,N1,N3], dtype=np.float32)
    np4 = np.zeros([N0,N3], dtype=np.float32)
    for ind1 in range(N0):
        np_h = hidden_b[ind1]
        for ind2 in range(sequence_length[ind1]):
            tmp1 = np.concatenate([inputx[ind1,sequence_length[ind1]-1-ind2],np_h], axis=0)
            np_h = hf_tanh(np.matmul(tmp1, kernel_b) + bias_b)
            np2[ind1,sequence_length[ind1]-1-ind2] = np_h
        np4[ind1] = np_h

    with tf.Graph().as_default() as tfG:
        tf_x = tf.constant(inputx)
        tf_sequence_length = tf.constant(sequence_length)
        tf_hf = tf.constant(hidden_f)
        tf_hb = tf.constant(hidden_b)
        rnn0 = tf.nn.rnn_cell.BasicRNNCell(N3, name='rnn0')
        rnn1 = tf.nn.rnn_cell.BasicRNNCell(N3, name='rnn1')
        (tf1,tf2), (tf3,tf4) = tf.nn.bidirectional_dynamic_rnn(rnn0, rnn1, tf_x,
                initial_state_fw=tf_hf, initial_state_bw=tf_hb, sequence_length=tf_sequence_length, scope='bd')
        z1 = {x.name:x for x in rnn0.weights+rnn1.weights}
        aop = [tf.assign(z1['bd/fw/rnn0/kernel:0'], kernel_f), tf.assign(z1['bd/fw/rnn0/bias:0'], bias_f),
                tf.assign(z1['bd/bw/rnn1/kernel:0'], kernel_b), tf.assign(z1['bd/bw/rnn1/bias:0'], bias_b)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf1_,tf2_,tf3_,tf4_ = sess.run([tf1,tf2,tf3,tf4])
    print('tf_bidirectional_rnn fw:: np vs tf: ', hfe_r5(np1, tf1_))
    print('tf_bidirectional_rnn bw:: np vs tf: ', hfe_r5(np2, tf2_))
    print('tf_bidirectional_rnn fw hidden:: np vs tf: ', hfe_r5(np3, tf3_))
    print('tf_bidirectional_rnn bw hidden:: np vs tf: ', hfe_r5(np4, tf4_))


if __name__=='__main__':
    tf_rnn_single()
    print()
    tf_rnn_dropout()
    print()
    tf_rnn_sequence()
    print()
    tf_bidirectional_rnn()
