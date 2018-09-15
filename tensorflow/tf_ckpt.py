import os
import numpy as np
import tensorflow as tf

hf_data = lambda *x: os.path.join('..', 'data', *x)

# reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py
def tf_load_ckpt(filename):
    reader = tf.train.NewCheckpointReader(filename)
    return {k:reader.get_tensor(k) for k in reader.get_variable_to_shape_map()}

def _test_load_ckpt():
    z1 = tf_load_ckpt(hf_data('model.ckpt-15000'))
    for k,v in z1.items():
        print('{:10s} shape: {}'.format(k, v.shape))


def tf_get_group_aop():
    with tf.Graph().as_default() as tfG:
        _ = tf.Variable(0, dtype=tf.float32, name='tf') #'tf:0'
        _ = tf.Variable(1, dtype=tf.float32, name='tf') #'tf_1:0'

        variable_dict = {x.name:x for x in tfG.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        tmp1 = {'tf:0':1, 'tf_1:0':2}
        tmp2 = [tf.assign(variable_dict[x],y) for x,y in tmp1.items()]
        aop = tf.group(*tmp2, name='aop')
    print(aop)
    print(tfG.get_operation_by_name('aop'))


# TODO init_from_ckpt

if __name__=='__main__':
    _test_load_ckpt()
    print()
    tf_get_group_aop()


