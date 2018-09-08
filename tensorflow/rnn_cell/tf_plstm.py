import numpy as np
import tensorflow as tf

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)

hf_tanh = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
hf_sigmoid = lambda x: 1/(1+np.exp(-x))


def tf_plstm_single(N0=3, N2=7, N3=11, forget_bias=0.5, cell_clip=0.7):
    np1 = np.random.randn(N0, N2)
    np_c = np.random.randn(N0, N3)
    np_h = np.random.randn(N0, N3)
    np_w_f = np.random.randn(N3)
    np_w_i = np.random.randn(N3)
    np_w_o = np.random.randn(N3)
    np_kernel = np.random.randn(N2+N3, N3*4)
    np_bias = np.random.randn(N3*4)

    tmp1 = np.matmul(np.concatenate([np1,np_h], axis=1), np_kernel) + np_bias
    tmp2 = np.split(tmp1, 4, axis=1)
    tmp3 = hf_sigmoid(np_c*np_w_i + tmp2[0])*hf_tanh(tmp2[1])
    tmp4 = hf_sigmoid(tmp2[2] + np_w_f*np_c + forget_bias) * np_c
    np_c1 = tmp3 + tmp4
    np_c1 = np.clip(np_c1, -cell_clip, cell_clip)
    np_h1 = hf_tanh(np_c1) * hf_sigmoid(np_c1*np_w_o+tmp2[3])
    np2 = np_h1

    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1, name='input')
        tf_c = tf.constant(np_c, name='c')
        tf_h = tf.constant(np_h, name='h')
        plstm0 = tf.nn.rnn_cell.LSTMCell(N3, use_peepholes=True, cell_clip=cell_clip, forget_bias=forget_bias, name='plstm0')
        tf2,(tf_c1,tf_h1) = plstm0(tf1, (tf_c,tf_h))
        z1 = {x.name[7:-2]:x for x in plstm0.weights}
        aop = [
            tf.assign(z1['kernel'], np_kernel),
            tf.assign(z1['bias'], np_bias),
            tf.assign(z1['w_f_diag'], np_w_f),
            tf.assign(z1['w_i_diag'], np_w_i),
            tf.assign(z1['w_o_diag'], np_w_o),
        ]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf2_,tf_c1_,tf_h1_ = sess.run([tf2,tf_c1,tf_h1])

    print('tf_plstm_single output:: np vs tf: ', hfe_r5(np2, tf2_))
    print('tf_plstm_single h1:: np vs tf: ', hfe_r5(np_h1, tf_h1_))
    print('tf_plstm_single c1:: np vs tf: ', hfe_r5(np_c1, tf_c1_))


if __name__=='__main__':
    tf_plstm_single()
