from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def projection_layer(inputs, output_dim, name='', reuse=None, 
                    activation=None, weights_regularizer=None,
                    initializer=None, dropout=None, use_mode='FC',
                    num_layers=2, mode=''):
    """ Simple Projection layer 

    Args:
        x: `tensor`. vectors to be projected
            Shape is [batch_size x time_steps x emb_size]
        output_dim: `int`. dimensions of input embeddings
        rname: `str`. variable scope name
        reuse: `bool`. whether to reuse parameters within same
            scope
        activation: tensorflow activation function
        initializer: initializer
        dropout: dropout placeholder
        use_fc: `bool` to use fc layer api or matmul
        num_layers: `int` number layers of projection 
       
    Returns:
        A 3D `Tensor` of shape [batch, time_steps, output_dim]
    """
    # input_dim = tf.shape(inputs)[2]
    if(initializer is None):
        initializer = tf.contrib.layers.xavier_initializer()
    input_dim = inputs.get_shape().as_list()[2]
    time_steps = tf.shape(inputs)[1]
    with tf.variable_scope('projection_{}'.format(name), reuse=reuse) as scope:
        x = tf.reshape(inputs, [-1, input_dim])
        output = x
        for i in range(num_layers):
            if(dropout is not None):
                output = tf.nn.dropout(output, dropout)
            _dim = output.get_shape().as_list()[1]
            if(use_mode=='FC'):
                weights = tf.get_variable('weights_{}'.format(i),
                              [_dim, output_dim], 
                              initializer=initializer) 
                zero_init = tf.zeros_initializer()
                bias = tf.get_variable('bias_{}'.format(i), shape=output_dim,
                                            dtype=tf.float32,
                                            initializer=zero_init)
                output = tf.nn.xw_plus_b(output, weights, bias)
            elif(use_mode=='HIGH'):
                output = highway_layer(output, output_dim, initializer, 
                                name='proj_{}'.format(i), reuse=reuse)
            else:
                weights = tf.get_variable('weights_{}'.format(i),
                              [_dim, output_dim], 
                              initializer=initializer) 
                output = tf.matmul(output, weights)
            if(activation is not None and use_mode!='HIGH'):
                output = activation(output)
            
        
        output = tf.reshape(output, [-1, time_steps, output_dim])
        
        
        return output

        
def highway_layer(input_data, dim, init, name='', reuse=None):
    """ Creates a highway layer
    """
    trans = linear(input_data, dim, init,  name='trans_{}'.format(name), 
                        reuse=reuse)
    trans = tf.nn.relu(trans)
    gate = linear(input_data, dim, init, name='gate_{}'.format(name), 
                        reuse=reuse)
    gate = tf.nn.sigmoid(gate)
    if(dim!=input_data.get_shape()[-1]):
        input_data = linear(input_data, dim, init,name='trans2_{}'.format(name),
                            reuse=reuse)
    output = gate * trans + (1-gate) * input_data
    return output

def linear(input_data, dim, initializer, name='', reuse=None):
    """ Default linear layer
    """
    input_shape = input_data.get_shape().as_list()[1]
    with tf.variable_scope('linear', reuse=reuse) as scope:
        _weights = tf.get_variable(
                "W_{}".format(name),
                shape=[input_shape, dim],
                initializer=initializer)
        _bias = tf.get_variable('bias_{}'.format(name),
                shape=[dim],
                initializer=tf.constant_initializer([0.1]))
    output_data = tf.nn.xw_plus_b(input_data, _weights, _bias)
    return output_data


def mask_zeros_1(embed, lens, max_len, expand=True):
    mask = tf.sequence_mask(lens, max_len)
    mask = tf.cast(mask, tf.float32)
    if(expand):
        mask = tf.expand_dims(mask, 2)
    embed = embed * mask
    return embed

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