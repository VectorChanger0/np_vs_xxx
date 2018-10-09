import random
import string
import tensorflow as tf


def _generate_string():
    len_ = [random.randint(3,7) for _ in range(random.randint(3,7))]
    ret = ' '.join(''.join(random.choices(string.ascii_lowercase, k=k)) for k in len_)
    return ret

def tf_string_split():
    N0 = 3
    inputx = [_generate_string() for _ in range(N0)]
    ret1 = [x.split(' ') for x in inputx]

    with tf.Graph().as_default() as tfG:
        tf_x = tf.constant(inputx)
        tf1 = tf.string_split(tf_x)
        sequence_length = tf.bincount(tf.cast(tf1.indices[:,0], tf.int32), dtype=tf.int64)
        tf2 = tf.sparse_tensor_to_dense(tf1, '')
    with tf.Session(graph=tfG) as sess:
        tf2_,sequence_length_ = sess.run([tf2,sequence_length])
        ret2 = [[x.decode() for x in tf2_[ind1,:sequence_length_[ind1]]] for ind1 in range(tf2_.shape[0])]
    print('tf_string_split:: str vs tf: ', ret1==ret2)


if __name__=='__main__':
    tf_string_split()
