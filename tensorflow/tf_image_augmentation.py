import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

'''
image(tf,float,(H,W,3))
(ret):image(tf,float,(H,W,3))

for large enough image, BILINEAR or NEAREST should not make big differences
'''

hf_data = lambda *x: os.path.join('..','data',*x)

def tf_image_rotate(image, theta):
    return tf.contrib.image.rotate(image, theta, interpolation='NEAREST')

def tf_image_translate(image, dh, dw):
    return tf.contrib.image.translate(image, [dw,dh], interpolation='NEAREST')

def tf_image_shear_x(image, theta):
    tmp1 = [1, tf.sin(theta),0,0,tf.cos(theta),0,0,0]
    return tf.contrib.image.transform(image, tmp1, interpolation='NEAREST')

def tf_image_shear_y(image, theta):
    tmp1 = [tf.cos(theta),0,0,tf.sin(theta),1,0,0,0]
    return tf.contrib.image.transform(image, tmp1, interpolation='NEAREST')

def tf_image_scale(image, ratio):
    H,W,C = image.shape.as_list()
    H1 = tf.cast(H*ratio, tf.int32)
    W1 = tf.cast(W*ratio, tf.int32)
    image = tf.image.resize_images(image, (H1,W1))
    scale_up = lambda: tf.random_crop(image, [H,W,C])
    def scale_down():
        tmp1 = tf.random_uniform([],0,H-H1+1,dtype=tf.int32)
        tmp2 = tf.random_uniform([],0,W-W1+1,dtype=tf.int32)
        tmp3 = [[tmp1,H-H1-tmp1],[tmp2,W-W1-tmp2],[0,0]]
        return tf.pad(image, tmp3)
    ret = tf.cond(ratio>1, scale_up, scale_down)
    ret.set_shape((H,W,C))
    return ret

def tf_image_augmentation(image, flip_x=True, flip_y=False, scale=(0.9,1.1), rotate=(-0.1,0.1),
        shear_x=(-0.1,0.1), shear_y=(-0.1,0.1), translate_ratio=(-0.1,0.1), name='augmentation'):
    '''
    image(tf,float32,(H,W,C))
    flip_x/flip_y(bool)
    scale(tuple/float): in normalized units
    rotate/shear_x/shear_y(tuple/float): in radian
    translate_ratio(tuple/float): in normalized units
    name(str)
    (ret)(tf,float32,(H,W,C))
    '''
    with tf.variable_scope(name):
        H,W,C = image.shape.as_list()
        if flip_x: image = tf.image.random_flip_left_right(image)
        if flip_y: image = tf.image.random_flip_up_down(image)

        # scale
        ratio = tf.random_uniform([], scale[0], scale[1])
        H1,W1 = tf.cast(H*ratio, tf.int32), tf.cast(W*ratio, tf.int32)
        image = tf.image.resize_images(image, (H1,W1))
        scale_up = lambda: tf.random_crop(image, [H,W,C])
        def scale_down():
            tmp1 = tf.random_uniform([],0,H-H1+1,dtype=tf.int32)
            tmp2 = tf.random_uniform([],0,W-W1+1,dtype=tf.int32)
            tmp3 = [[tmp1,H-H1-tmp1],[tmp2,W-W1-tmp2],[0,0]]
            return tf.pad(image, tmp3)
        image = tf.cond(ratio>1, scale_up, scale_down)
        image.set_shape((H,W,C))

        # rotate
        theta = tf.random_uniform([], rotate[0], rotate[1])
        image = tf.contrib.image.rotate(image, theta, interpolation='NEAREST')

        # shear_x
        theta = tf.random_uniform([], shear_x[0], shear_x[1])
        tmp1 = [1, tf.sin(theta),0,0,tf.cos(theta),0,0,0]
        image = tf.contrib.image.transform(image, tmp1, interpolation='NEAREST')

        # shear_y
        theta = tf.random_uniform([], shear_y[0], shear_y[1])
        tmp1 = [tf.cos(theta),0,0,tf.sin(theta),1,0,0,0]
        image = tf.contrib.image.transform(image, tmp1, interpolation='NEAREST')

        # translate
        dh = tf.random_uniform([], int(H*translate_ratio[0]), int(H*translate_ratio[1]))
        dw = tf.random_uniform([], int(W*translate_ratio[0]), int(W*translate_ratio[1]))
        image = tf.contrib.image.translate(image, [dw,dh], interpolation='NEAREST')
        return image


def _show_image(np1, title):
    plt.figure()
    plt.imshow(np1/255)
    plt.title(title)

if __name__=='__main__':
    filename = hf_data('coffee.jpg')
    with Image.open(filename) as image_pil:
        np1 = np.array(image_pil, dtype=np.float32)
        H,W,_ = np1.shape
    tf1 = tf.constant(np1, dtype=tf.float32)
    tf2 = tf_image_rotate(tf1, tf.random_uniform([], -7*np.pi/180, 7*np.pi/180))
    tf3 = tf_image_translate(tf1, tf.random_uniform([], -int(H*0.1), int(H*0.1)), tf.random_uniform([], -int(W*0.1), int(W*0.1)))
    tf4 = tf_image_shear_x(tf1, tf.random_uniform([], -7*np.pi/180, 7*np.pi/180))
    tf5 = tf_image_shear_y(tf1, tf.random_uniform([], -7*np.pi/180, 7*np.pi/180))
    tf6 = tf_image_scale(tf1, tf.random_uniform([], 0.9, 1.1))
    tf7 = tf_image_augmentation(tf1)
    with tf.Session() as sess:
        tf2_,tf3_,tf4_,tf5_,tf6_,tf7_ = sess.run([tf2,tf3,tf4,tf5,tf6,tf7])

    _show_image(np1, 'origin')
    _show_image(tf2_, 'random rotate')
    _show_image(tf3_, 'random translate')
    _show_image(tf4_, 'random shear x')
    _show_image(tf5_, 'random shear y')
    _show_image(tf6_, 'random scale')
    _show_image(tf7_, 'augmentation')
    plt.show()

