import os
import cv2
import numpy as np
from PIL import Image
from skimage import io
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize

hf_data = lambda *x: os.path.join('..', 'data',*x)
hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)

'''
TensorFlow strange resize method

mapping relation for an image of size (h,w) -> (h',w')
image index | skimage           | tensorflow         | tf.pad([[1,1].[1,1]], 'SYMMETRIC')
(0,0)       | (1/2h,1/2w)       | (0,0)              | (1/(h+1),1/(w+1))
(1,1)       | (3/2h,3/2w)       | (1/(h-1),1/(w-1))  | (2/(h+1),2/(w+1))
(h-1,w-1)   | (1-1/2h,1-1/2w)   | (1,1)              | (h/(h+1),w/(w+1))
(0',0')     | (1/2h',1/2w')     | nan                | (h/h'+1)/(2h+2), (w/w'+1)/(2w+2) [[1]]
(1',1')     | (3/2h',3/2w')     | (3h/h'-1)/2(h-1)   | TBA
(h'-1,w'-1) | (1-1/2h',1-1/2w') | nan                | (2*h+1-h/h')/(2h+2), (2*w+1-w/w')/(2w+2) [[2]]
'''


def tf_image_resize_bilinear(image, size, name='resize_bilinear'):
    '''
    # reference: https://www.zhihu.com/question/286255813/answer/449907307
    recommand not to use tf.image.resize_bilinear(align_corner=True/False)

    image(tf,?,(?,h0,w0,?))
    size(tuple,int,(2,))
    name(str)
    '''
    h1,w1 = size
    with tf.variable_scope(name):
        h0,w0 = image.shape.as_list()[1:3]
        N0 = tf.shape(image)[0]
        image = tf.pad(image, [[0,0],[1,1],[1,1],[0,0]], mode='SYMMETRIC')
        nh1 = (h0/h1 + 1)/(2*h0+2) # see mapping relation [[1]]
        nw1 = (w0/w1 + 1)/(2*w0+2)
        nh2 = (2*h0+1-h0/h1)/(2*h0+2) # see mapping relation [[2]]
        nw2 = (2*w0+1-w0/w1)/(2*w0+2)
        tmp1 = tf.tile(np.array([[nh1,nw1,nh2,nw2]], dtype=np.float32), [N0,1])
        tmp2 = tf.range(N0)
        return tf.image.crop_and_resize(image, tmp1, tmp2, [h1,w1])

def _test_tf_image_resize_bilinear(N0=2, N1=3, N2=5, N3=7, N4=11, C1=13):
    np1 = np.random.uniform(0,1,size=(N0,N1,N2,C1)).astype(np.float32)
    np2 = np.array([resize(x, [N3,N4], mode='symmetric', anti_aliasing=False) for x in np1])

    tf1 = tf.constant(np1)
    tf2 = tf_image_resize_bilinear(tf1, (N3,N4))
    with tf.Session() as sess:
        tf2_ = sess.run(tf2)
    print('tf_resize_bilinear:: skimage vs tf: ', hfe_r5(np2,tf2_))


def tf_image_crop_and_resize(image, pbox, box_ind, crop_size, name='crop_and_resize'):
    '''
    # reference: https://www.zhihu.com/question/286255813/answer/449907307
    recommand not to use tf.image.crop_and_resize()
    image(tf,?,(?,h0,w0,?))
    pbox(tf,float,(?,4))
    crop_size(tuple,int,(2,))
    name(str)
    '''
    with tf.variable_scope(name):
        H0,W0 = image.shape.as_list()[1:3]
        H2,W2 = crop_size
        N0 = tf.shape(image)[0]
        h1,w1,h2,w2 = tf.split(pbox, 4, axis=1)
        tmp1 = (h2-h1)/(2*H2)
        tmp2 = (w2-w1)/(2*W2)
        h1 = (h1+0.5+tmp1)/(H0+1)
        h2 = (h2+0.5-tmp1)/(H0+1)
        w1 = (w1+0.5+tmp2)/(W0+1)
        w2 = (w2+0.5-tmp2)/(W0+1)
        boxes = tf.concat([h1,w1,h2,w2], axis=1)
        image = tf.pad(image, [[0,0],[1,1],[1,1],[0,0]], mode='SYMMETRIC')
        return tf.image.crop_and_resize(image, boxes, box_ind, crop_size)

def _test_tf_image_crop_and_resize(C1=3, N1=5, N2=7):
    # crop to origin size
    np1 = np.random.rand(1, N1, N2, C1).astype(np.float32)
    tmp1 = np.sort(np.random.choice(N1, 2, False))
    tmp2 = np.sort(np.random.choice(N2, 2, False))
    pbox = np.array([[tmp1[0],tmp2[0],tmp1[1],tmp2[1]]], dtype=np.float32)
    np2 = np1[:,tmp1[0]:tmp1[1], tmp2[0]:tmp2[1],:]
    tf1 = tf.constant(np1)
    tf2 = tf.constant(pbox)
    tf3 = tf_image_crop_and_resize(tf1, tf2, [0], [tmp1[1]-tmp1[0],tmp2[1]-tmp2[0]])
    with tf.Session() as sess:
        tf3_ = sess.run(tf3)
    print('tf_image_crop_and_resize_0:: np vs tf: ', hfe(np2, tf3_))

    # example reference: https://www.zhihu.com/question/286255813/answer/449907307
    np1 = np.arange(25).reshape((5,5)).astype(np.float32)
    np2 = np.array([[4.5,5,5.5,6],[7,7.5,8,8.5],[9.5,10,10.5,11],[12,12.5,13,13.5]], dtype=np.float32)
    tf1 = tf.constant(np1[np.newaxis,:,:,np.newaxis])
    tf2 = tf.constant([[1,1,3,3]], dtype=tf.float32)
    tf3 = tf_image_crop_and_resize(tf1, tf2, [0], [4,4])
    with tf.Session() as sess:
        tf3_ = sess.run(tf3)[0,:,:,0]
    print('tf_image_crop_and_resize_1:: np vs tf: ', hfe_r5(np2, tf3_))

    ## TODO crop_and_resize, the number in the center should match with skimage results
    ## ....bu hui xie....


def tf_image_crop_and_resize_gradients(N1=5, N2=7, C1=3, eps=1e-5):
    ind1 = np.random.randint(0,4,[])
    np1 = np.random.uniform(0,1,size=(1,N1,N2,C1))
    np2 = np.array([[np.random.uniform(0.1,0.3,[]), np.random.uniform(0.1,0.3,[]),
            np.random.uniform(0.7,0.9,[]),np.random.uniform(0.7,0.9,[])]])
    tf1 = tf.constant(np1, dtype=tf.float32)
    tf2 = tf.constant(np2, dtype=tf.float32)
    tf3 = tf.reduce_sum(tf.image.crop_and_resize(tf1, tf2, [0], [5,5]))
    tfg1 = tf.gradients(tf3, tf2)[0]
    with tf.Session() as sess:
        tfg1_ = sess.run(tfg1)

    with tf.Session() as sess:
        tmp1 = np2.copy()
        tmp1[0,ind1] += eps
        tmp2 = np2.copy()
        tmp2[0,ind1] -= eps
        npg1 = (sess.run(tf3, feed_dict={tf2:tmp1}) - sess.run(tf3, feed_dict={tf2:tmp2}))/2/eps
    print('tf_image_crop_and_resize_gradients:: tf vs tf: ', hfe_r5(tfg1_[0,ind1], npg1))


def tf_load_image_accurate():
    print('warning: results in wsl is different from anaconda prompt')
    tmp0 = hf_data('coffee.jpg')
    np1 = io.imread(tmp0) #type1
    np2 = cv2.imread(tmp0)[:,:,::-1] #type1

    tf0 = tf.read_file(tmp0)
    tf1 = tf.image.decode_jpeg(tf0, channels=3) #type2
    tf2 = tf.image.decode_jpeg(tf0, channels=3, dct_method='INTEGER_ACCURATE') #type1
    tf3 = tf.image.decode_jpeg(tf0, dct_method='INTEGER_ACCURATE') #type1
    tf4 = tf.image.decode_image(tf0) #type2
    with tf.Session() as sess:
        tf1_,tf2_,tf3_,tf4_ = sess.run([tf1, tf2, tf3, tf4])
    print('PIL as stand results')
    np1 = np1.astype(np.float32)
    print('cv2 relative error: ', hfe_r5(np1,np2))
    print('tf.image.decode_jpeg(3): ', hfe_r5(np1, tf1_))
    print('tf.image.decode_jpeg(3, "INTEGER_ACCURATE"): ', hfe_r5(np1,tf2_))
    print('tf.image.decode_jpeg("INTEGER_ACCURATE"): ', hfe_r5(np1, tf3_))
    print('tf.image.decode_image: ', hfe_r5(np1, tf4_))


def PIL_scale_crop(filename, h0, w0):
    with Image.open(filename) as image_pil:
        h1,w1 = image_pil.height, image_pil.width
        scale = max(h0/h1, w0/w1)
        h2,w2 = round(h1*scale),round(w1*scale)
        h3,w3 = (h2-h0)//2, (w2-w0)//2
        tmp1 = image_pil.resize((w2,h2), resample=Image.BILINEAR).crop((w3,h3,w3+w0,h3+h0))
        return np.array(tmp1, dtype=np.uint8) #(np,uint8,(h0,w0,3))

def _test_PIL_scale_crop():
    plt.figure()
    plt.imshow(PIL_scale_crop(hf_data('coffee.jpg'), 800,800))
    plt.title('PIL_scale_crop')


def PIL_scale_pad(filename, h0, w0):
    with Image.open(filename) as image_pil:
        h1,w1 = image_pil.height, image_pil.width
        scale = min(h0/h1, w0/w1)
        h2,w2 = round(h1*scale),round(w1*scale)
        h3,w3 = (h2-h0)//2, (w2-w0)//2
        tmp1 = image_pil.resize((w2,h2), resample=Image.BILINEAR).crop((w3,h3,w3+w0,h3+h0))
        return np.array(tmp1, dtype=np.uint8) #(np,uint8,(h0,w0,3))

def _test_PIL_scale_pad():
    plt.figure()
    plt.imshow(PIL_scale_pad(hf_data('coffee.jpg'), 800,800))
    plt.title('PIL_scale_pad')


def tf_scale_crop(filename, h0_i, w0_i, data_format):
    # filename(str)(tf,str,(,))
    # h0(int)
    # w0(int)
    # data_format(str)
    # (ret)(tf,float32,(h0_i,w0_i,3))
    h0 = tf.constant(h0_i, dtype=tf.float32)
    w0 = tf.constant(w0_i, dtype=tf.float32)
    
    tmp0 = tf.read_file(filename)
    image = tf.cast(tf.image.decode_jpeg(tmp0, channels=3, dct_method='INTEGER_ACCURATE'), tf.float32)
    tmp0 = tf.cast(tf.shape(image), tf.float32)
    h1,w1 = tmp0[0],tmp0[1]

    scale = tf.maximum(h0/h1,w0/w1)
    h2,w2 = tf.round(h1*scale),tf.round(w1*scale)
    h2_i,w2_i = tf.cast(h2,tf.int32), tf.cast(w2,tf.int32)
    h3_i = tf.cast(tf.floor((h2-h0)/2), tf.int32)
    w3_i = tf.cast(tf.floor((w2-w0)/2), tf.int32)

    image = tf.image.resize_images(image, [h2_i,w2_i])[h3_i:(h3_i+h0_i),w3_i:(w3_i+w0_i),:]
    if data_format=='channels_first':
        image = tf.transpose(image, [2,0,1])
        image.set_shape((3,h0_i,w0_i))
    else:
        image.set_shape((h0_i,w0_i,3))
    return image

def _test_tf_scale_crop():
    tf1 = tf_scale_crop(hf_data('coffee.jpg'), 800, 800, 'channels_last')
    with tf.Session() as sess:
        tf1_ = sess.run(tf1)
    plt.figure()
    plt.imshow(tf1_/255)
    plt.title('tf_scale_crop')


def tf_scale_pad(filename, h0_i, w0_i, data_format):
    # filename(str)(tf,str,(,))
    # h0(int)
    # w0(int)
    # data_format(str)
    # (ret)(tf,float32,(h0_i,w0_i,3))
    h0 = tf.constant(h0_i, dtype=tf.float32)
    w0 = tf.constant(w0_i, dtype=tf.float32)
    
    tmp0 = tf.read_file(filename)
    image = tf.cast(tf.image.decode_jpeg(tmp0, channels=3, dct_method='INTEGER_ACCURATE'), tf.float32)
    tmp0 = tf.cast(tf.shape(image), tf.float32)
    h1,w1 = tmp0[0],tmp0[1]

    scale = tf.minimum(h0/h1,w0/w1)
    h2,w2 = tf.round(h1*scale),tf.round(w1*scale)
    h2_i,w2_i = tf.cast(h2,tf.int32), tf.cast(w2,tf.int32)
    h3_i = tf.cast(tf.floor((h0-h2)/2), tf.int32)
    w3_i = tf.cast(tf.floor((w0-w2)/2), tf.int32)
    tmp1 = [(h3_i,h0_i-h3_i-h2_i),(w3_i,w0_i-w3_i-w2_i),(0,0)]

    image = tf.pad(tf.image.resize_images(image, [h2_i,w2_i]), tmp1)
    if data_format=='channels_first':
        image = tf.transpose(image, [2,0,1])
        image.set_shape((3,h0_i,w0_i))
    else:
        image.set_shape((h0_i,w0_i,3))
    return image

def _test_tf_scale_pad():
    tf1 = tf_scale_pad(hf_data('coffee.jpg'), 800, 800, 'channels_last')
    with tf.Session() as sess:
        tf1_ = sess.run(tf1)
    plt.figure()
    plt.imshow(tf1_/255)
    plt.title('tf_scale_pad')


if __name__=='__main__':
    _test_tf_image_resize_bilinear()
    print('')
    _test_tf_image_crop_and_resize()
    print('')
    tf_image_crop_and_resize_gradients()
    print('')
    tf_load_image_accurate()
    print('')
    _test_PIL_scale_crop()
    print('')
    _test_PIL_scale_pad()
    print('')
    _test_tf_scale_crop()
    print('')
    _test_tf_scale_pad()
    print('')

    plt.show()

