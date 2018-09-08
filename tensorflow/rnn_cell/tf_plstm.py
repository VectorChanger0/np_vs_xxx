import numpy as np
import tensorflow as tf

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)

hf_tanh = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
hf_sigmoid = lambda x: 1/(1+np.exp(-x))


def tf_plstm_single(N0=3, N2=7, N3=11, N4=13, forget_bias=0.5, cell_clip=0.7, proj_clip=0.9):
    inputx = np.random.randn(N0, N2)
    cell = np.random.randn(N0, N3)
    hidden = np.random.randn(N0, N4)
    pf_kernel = np.random.randn(N3)
    pu_kernel = np.random.randn(N3)
    po_kernel = np.random.randn(N3)
    ucfo_kernel = np.random.randn(N2+N4, N3*4)
    ucfo_bias = np.random.randn(N3*4)
    project_kernel = np.random.randn(N3,N4)

    tmp1 = np.concatenate([inputx, hidden], axis=1)
    Du,Dc,Df,Do = np.split(np.matmul(tmp1, ucfo_kernel) + ucfo_bias, 4, axis=1)
    tmp1 = hf_sigmoid(Du + cell*pu_kernel)*hf_tanh(Dc)
    tmp2 = hf_sigmoid(Df + pf_kernel*cell + forget_bias) * cell
    np_c1 = np.clip(tmp1 + tmp2, -cell_clip, cell_clip)
    np_h1 = hf_sigmoid(Do + np_c1*po_kernel) * hf_tanh(np_c1)
    np_h1 = np.clip(np.matmul(np_h1, project_kernel), -proj_clip, proj_clip)
    np1 = np_h1

    with tf.Graph().as_default() as tfG:
        tf_x = tf.constant(inputx, name='inputx')
        tf_c = tf.constant(cell, name='cell')
        tf_h = tf.constant(hidden, name='hidden')
        plstm0 = tf.nn.rnn_cell.LSTMCell(N3, use_peepholes=True, cell_clip=cell_clip, 
                num_proj=N4, proj_clip=proj_clip, forget_bias=forget_bias, name='plstm0')
        tf1,(tf_c1,tf_h1) = plstm0(tf_x, (tf_c,tf_h))
        z1 = {x.name[7:-2]:x for x in plstm0.weights}
        aop = [
            tf.assign(z1['kernel'], ucfo_kernel),
            tf.assign(z1['bias'], ucfo_bias),
            tf.assign(z1['w_f_diag'], pf_kernel),
            tf.assign(z1['w_i_diag'], pu_kernel),
            tf.assign(z1['w_o_diag'], po_kernel),
            tf.assign(z1['projection/kernel'], project_kernel),
        ]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        tf1_,tf_c1_,tf_h1_ = sess.run([tf1,tf_c1,tf_h1])

    print('tf_lstm_peep_single output:: np vs tf: ', hfe_r5(np1, tf1_))
    print('tf_lstm_peep_single h1:: np vs tf: ', hfe_r5(np_h1, tf_h1_))
    print('tf_lstm_peep_single c1:: np vs tf: ', hfe_r5(np_c1, tf_c1_))


if __name__=='__main__':
    tf_plstm_single()
