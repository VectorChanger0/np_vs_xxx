import os
import numpy as np
import tensorflow as tf

from utils import next_tbd_dir

# TODO use hook to restore weights

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)


def compute_graph(x, mode):
    training= mode==tf.estimator.ModeKeys.TRAIN
    x = tf.layers.dense(x, 5, name='fc1')
    x = tf.layers.batch_normalization(x, training=training, name='bn1')
    x = tf.nn.relu(x)
    x = tf.layers.dense(x, 5, name='fc2')
    x = tf.nn.relu(x)
    logits = tf.layers.dense(x, 1, name='logits')[:,0]
    return logits

def load_ckpt(filename):
    reader = tf.train.NewCheckpointReader(filename)
    return {k:reader.get_tensor(k) for k in reader.get_variable_to_shape_map()}


'''parameters'''
logdir1 = next_tbd_dir()
hf_file1 = lambda *x: os.path.join(logdir1, *x)
logdir2 = next_tbd_dir()
hf_file2 = lambda *x: os.path.join(logdir2, *x)
np_dict = {}


'''first build compute graph, initialize with some random number and save to logdir1'''
with tf.Graph().as_default() as tfG:
    x = tf.placeholder(dtype=tf.float32, name='features', shape=(None,3))
    _ = compute_graph(x, tf.estimator.ModeKeys.TRAIN)

    z1 = tfG.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    aop = []
    for x in tfG.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        tmp1 = np.random.uniform(size=x.shape.as_list())
        np_dict[x.name] = tmp1
        aop += [tf.assign(x, tmp1)]
    saver = tf.train.Saver()

with tf.Session(graph=tfG) as sess:
    _ = sess.run(aop)
    saver.save(sess, hf_file1('model.ckpt'), global_step=0)

'''second, build compute graph with estimator api and resotre part of parameter from logdir1'''
def model_fn(features, labels, mode, params):
    logits = compute_graph(features, mode)
    
    tmp1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels,tf.float32), logits=logits)
    loss = tf.reduce_mean(tmp1, name='loss')
    global_step = tf.train.get_or_create_global_step()
    if 'ckpt' in params:
        tf.train.init_from_checkpoint(params['ckpt'], params['assignment_map'])
        train_op = tf.train.GradientDescentOptimizer(1e-300).minimize(loss, global_step)
    else:
        optimizer = tf.train.GradientDescentOptimizer(params['lr'])
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

npX = np.random.rand(100,3).astype(np.float32)
npy = np.random.randint(0, 2, [100])
ds_train = lambda: tf.data.Dataset.from_tensor_slices((npX,npy)).repeat().batch(1)

ckpt = hf_file1('model.ckpt-0')
tmp1 = {k:k for k in tf.train.NewCheckpointReader(ckpt).get_variable_to_shape_map() if k[:6]!='logits'}
params = {'ckpt':ckpt, 'assignment_map':tmp1, 'lr':0.1}
train_config = tf.estimator.RunConfig(save_checkpoints_steps=1000, keep_checkpoint_max=10)
DNN = tf.estimator.Estimator(model_fn, logdir2, train_config, params)
DNN.train(ds_train, steps=1)


'''third, check difference between two ckpt'''
z1 = load_ckpt(tf.train.latest_checkpoint(logdir2))
z2 = {k[:-2]:v for k,v in np_dict.items()}
print('notice logits is not init from ckpt, and global_step is not in ckpt1')
for k in z2.keys():
    print('init_from_ckpt:: {}: {}'.format(k, hfe_r5(z1[k],z2[k])))
