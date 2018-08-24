# TODO 

# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

# tf.enable_eager_execution()
# tfe = tf.contrib.eager

# hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
# hfe_r5 = lambda x,y: round(hfe(x,y), 5)

# def grad(f): return lambda x: tfe.gradients_function(f)(x)[0]

# def hf1(x): return tf.square(tf.sin(x))
# grad_hf1 = grad(hf1)

# print(hfe_r5(hf1(np.pi/2).numpy(), 1))
# print(hfe_r5(grad_hf1(np.pi/2).numpy(), 0))

# # higher-order gradients
# x = tf.lin_space(-2*np.pi, 2*np.pi, 100)
# plt.plot(x, hf1(x), label="f")
# plt.plot(x, grad(hf1)(x), label="first derivative")
# plt.plot(x, grad(grad(hf1))(x), label="second derivative")
# plt.plot(x, grad(grad(grad(hf1)))(x), label="third derivative")
# plt.legend()
# plt.show()

