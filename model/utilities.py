from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def hyperbolic_ball(x, y, neg=False, eps=1E-6):
    """ Poincare Distance Function.
    """
    z = x - y
    z = tf.norm(z, ord='euclidean', keep_dims=True, axis=1)
    z = tf.square(z)
    x_d = 1 - tf.square(tf.norm(x, ord='euclidean',
                                    keep_dims=True, 
                                    axis=1))
    y_d = 1 - tf.square(tf.norm(y, ord='euclidean',
                                    keep_dims=True, 
                                    axis=1))
    d = x_d * y_d
    z = z / (d + eps)
    z  = (2 * z) + 1
    arcosh = z + tf.sqrt(tf.square(z - 1) + eps)
    arcosh = tf.log(arcosh)
    if(neg):
        arcosh = -arcosh
    return arcosh

def H2E_ball(grad, eps=1E-5):
    ''' Converts hyperbolic gradient to euclidean gradient
    '''
    # print(grad)
    if(grad is None):
        return None
    try:
        shape = grad.get_shape().as_list()
        if(len(shape)>=3):
            grad_scale = 1 - tf.square(tf.norm(grad, axis=[-2,-1], 
                                    ord='euclidean', keep_dims=True))
        elif(len(shape)==2):
            grad_scale = 1 - tf.square(tf.norm(grad, ord='euclidean', 
                                    keep_dims=True))
        else:
            return grad
    except:
        grad_scale = 1 - tf.square(tf.norm(grad, ord='euclidean', 
                                    keep_dims=True))

    grad_scale = tf.square(grad_scale) + eps
    grad_scale = (grad_scale) / 4
    grad = grad * grad_scale
    # grad = tf.clip_by_norm(grad, 1.0, axes=0)
    return grad

def clip_sentence(sentence, sizes):
    """ Clip the input sentence placeholders to the length of 
        the longest one in the batch. This saves processing time.

    Args:
        sentence: `tensor`shape (batch, time_steps)
        sizes `tensor` shape (batch)

    Return: 
        clipped_sent: `tensor` with shape (batch, time_steps)
    """

    max_batch_size = tf.reduce_max(sizes)
    clipped_sent = tf.slice(sentence, [0, 0],
                            tf.stack([-1, max_batch_size]))
    # clipped_sent = tf.reshape(clipped_sent, [-1, max_batch_size])
    return clipped_sent, max_batch_size

def mean_over_time(inputs, lengths):
    ''' Implements a MoT layer.

    Takes average vector across temporal dimension.

    Args:
        inputs: `tensor` [bsz x timestep x dim]
        lengths: `tensor` [bsz x 1] of sequence lengths

    Returns:
        mean_vec:`tensor` [bsz x dim]
    '''
    mean_vec = tf.reduce_sum(inputs, 1) 
    mean_vec = tf.div(mean_vec, tf.cast(lengths, tf.float32))
    return mean_vec