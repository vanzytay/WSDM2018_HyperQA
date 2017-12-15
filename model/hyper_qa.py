#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
import gzip
import json
from tqdm import tqdm
import random
from collections import Counter
import operator
import timeit
import time
import datetime
from keras.utils import np_utils
import numpy as np

from .utilities import *

class HyperQA:
    ''' Hyperbolic Embeddings for QA
    '''
    def __init__(self, vocab_size, args, char_vocab=0, pos_vocab=0):
        self.vocab_size = vocab_size
        self.char_vocab = char_vocab
        self.pos_vocab = pos_vocab
        self.graph = tf.Graph()
        self.args = args
        self.imap = {}
        self.inspect_op = []
        self.feat_prop = None
        if(self.args.init_type=='xavier'):
            self.initializer = tf.contrib.layers.xavier_initializer()
        elif(self.args.init_type=='normal'):
            self.initializer = tf.random_normal_initializer(0.0, 
                                                    self.args.init)
        elif(self.args.init_type=='uniform'):
            self.initializer = tf.random_uniform_initializer(
                                                    maxval=self.args.init, 
                                                    minval=-self.args.init)
        self.build_graph()
        
    def _get_pair_feed_dict(self, data, mode='training', lr=None):
        # print(data[0])
        data = zip(*data)
        labels = data[-1]

        if(lr is None):
            lr = self.args.learn_rate

        if(mode=='training'):
            # print(data[5])
            assert(np.min(data[1])>0)
            assert(np.min(data[3])>0)
            assert(np.min(data[5])>0)

        feed_dict = {
            self.q1_inputs:data[self.imap['q1_inputs']],
            self.q2_inputs:data[self.imap['q2_inputs']],
            self.q1_len:data[self.imap['q1_len']],
            self.q2_len:data[self.imap['q2_len']],
            self.q3_inputs:data[self.imap['q3_inputs']],
            self.q3_len:data[self.imap['q3_len']],
            self.learn_rate:lr,
            self.dropout:self.args.dropout,
            self.emb_dropout:self.args.emb_dropout
        }

        if(mode!='training'):
            feed_dict[self.dropout] = 1.0
            feed_dict[self.emb_dropout] = 1.0
        return feed_dict  

    def get_feed_dict(self, data, mode='training', lr=None):
        return self._get_pair_feed_dict(data, mode=mode, lr=lr)

    def register_index_map(self, idx, target):
        self.imap[target] = idx

    def build_glove(self, embed, lens, max_len):
        embed = mask_zeros_1(embed, lens, max_len)
        return tf.reduce_sum(embed, 1)

    def learn_repr(self, q1_embed, q2_embed, q1_len, q2_len, q1_max, 
                    q2_max, force_model=None, score=1, 
                    reuse=None, extract_embed=False,
                    side=''):

    
        translate_act = tf.nn.relu
        use_mode='FC'

        q1_embed = projection_layer(
                q1_embed,
                self.args.rnn_size,
                name='trans_proj',
                activation=translate_act,
                initializer=self.initializer,
                dropout=self.dropout,
                reuse=reuse,
                use_mode=use_mode,
                num_layers=self.args.num_proj
                )
        q2_embed = projection_layer(
                q2_embed,
                self.args.rnn_size,
                name='trans_proj',
                activation=translate_act,
                initializer=self.initializer,
                dropout=self.dropout,
                reuse=True,
                use_mode=use_mode,
                num_layers=self.args.num_proj
                )

        rnn_size = self.args.rnn_size
     
        q1_output = self.build_glove(q1_embed, q1_len, q1_max)
        q2_output = self.build_glove(q2_embed, q2_len, q2_max)
           
        try:
            self.max_norm = tf.reduce_max(tf.norm(q1_output, 
                                        ord='euclidean', 
                                        keep_dims=True, axis=1))
        except:
            self.max_norm = 0

        if(extract_embed):
            self.q1_extract = q1_output
            self.q2_extract = q2_output

        q1_output = tf.nn.dropout(q1_output, self.dropout)
        q2_output = tf.nn.dropout(q2_output, self.dropout)
        
        # This constraint is important
        _q1_output = tf.clip_by_norm(q1_output, 1.0, axes=1)
        _q2_output = tf.clip_by_norm(q2_output, 1.0, axes=1) 
         output = hyperbolic_ball(_q1_output, _q2_output)
       
        representation = output
        activation = None

        with tf.variable_scope('fl',reuse=reuse) as scope:
            last_dim = output.get_shape().as_list()[1]
            weights_linear = tf.get_variable('final_weights',
                            [last_dim, num_outputs],
                                initializer=self.initializer)
            bias_linear = tf.get_variable('bias', 
                                [num_outputs],
                                initializer=tf.zeros_initializer())

            final_layer = tf.nn.xw_plus_b(output, weights_linear, 
                                        bias_linear)
            output = final_layer

        return output, representation
  
    def build_graph(self):
        ''' Builds Computational Graph
        '''

        with self.graph.as_default():
            with tf.name_scope('q1_input'):
                self.q1_inputs = tf.placeholder(tf.int32, shape=[None, 
                                                    self.args.qmax], 
                                                    name='q1_inputs')
              
            with tf.name_scope('q2_input'):
                self.q2_inputs = tf.placeholder(tf.int32, shape=[None, 
                                                    self.args.amax], 
                                                    name='q2_inputs')
               
            with tf.name_scope('q3_input'):
                self.q3_inputs = tf.placeholder(tf.int32, shape=[None, 
                                                self.args.amax], 
                                                name='q3_inputs')
            with tf.name_scope('dropout'):
                self.dropout = tf.placeholder(tf.float32, name='dropout')
                self.emb_dropout = tf.placeholder(tf.float32, 
                                                name='emb_dropout')                 
            with tf.name_scope('q1_lengths'):
                self.q1_len = tf.placeholder(tf.int32, shape=[None])
        
            with tf.name_scope('q2_lengths'):
                self.q2_len = tf.placeholder(tf.int32, shape=[None])
                
            with tf.name_scope('q3_lengths'):
                self.q3_len = tf.placeholder(tf.int32, shape=[None])
               
            with tf.name_scope('learn_rate'):
                self.learn_rate = tf.placeholder(tf.float32, 
                                                name='learn_rate')

            if(self.args.pretrained==1):
                self.emb_placeholder = tf.placeholder(tf.float32, 
                            [self.vocab_size, self.args.emb_size])

            self.batch_size = tf.shape(self.q1_inputs)[0]
            
            q1_inputs, self.qmax = clip_sentence(self.q1_inputs, self.q1_len)
            q2_inputs, self.a1max = clip_sentence(self.q2_inputs, self.q2_len)
            q3_inputs, self.a2max = clip_sentence(self.q3_inputs, self.q3_len)

            with tf.variable_scope('embedding_layer'):
                if(self.args.pretrained==1):
                    self.embeddings = tf.Variable(tf.constant(
                                        0.0, shape=[self.vocab_size, 
                                            self.args.emb_size]), \
                                        trainable=self.args.trainable,
                                         name="embeddings")
                    self.embeddings_init = self.embeddings.assign(
                                        self.emb_placeholder)
                else:
                    self.embeddings = tf.Variable(tf.random_uniform(
                                    [self.vocab_size, self.args.emb_size], 
                                        -0.01, 0.01))

                q1_embed =  tf.nn.embedding_lookup(self.embeddings, 
                                                        q1_inputs)
                q2_embed =  tf.nn.embedding_lookup(self.embeddings, 
                                                        q2_inputs)
                q3_embed = tf.nn.embedding_lookup(self.embeddings, 
                                                        q3_inputs)
                        
            if(self.args.all_dropout):
                q1_embed = tf.nn.dropout(q1_embed, self.emb_dropout)
                q2_embed = tf.nn.dropout(q2_embed, self.emb_dropout)
                q3_embed = tf.nn.dropout(q3_embed, self.emb_dropout)


            compose_length = self.args.rnn_size
            rnn_length = self.args.rnn_size
            repr_fun = self.learn_repr

            self.output_pos, _  = repr_fun(q1_embed, q2_embed, 
                                        self.q1_len, self.q2_len, 
                                        self.qmax, 
                                        self.a1max, score=1, reuse=None, 
                                        extract_embed=True,
                                        side='POS',
                                        )

            self.output_neg,_ = repr_fun(q1_embed, 
                                        q3_embed, self.q1_len,
                                         self.q3_len, self.qmax, 
                                         self.a2max, score=1, 
                                         reuse=True, 
                                         side='NEG',
                                         )

            # Define loss and optimizer
            with tf.name_scope("train"):
                with tf.name_scope("cost_function"):
                     # hinge loss
                    self.hinge_loss = tf.maximum(0.0,(
                        self.args.margin - self.output_pos + self.output_neg))
                    
                    self.cost = tf.reduce_sum(self.hinge_loss)

                    with tf.name_scope('regularization'):
                        if(self.args.l2_reg>0):
                            vars = tf.trainable_variables() 
                            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars \
                                                if 'bias' not in v.name ])
                            lossL2 *= self.args.l2_reg
                            self.cost += lossL2
                   
                    tf.summary.scalar("cost_function", self.cost)
                global_step = tf.Variable(0, trainable=False)
                
                if(self.args.decay_lr>0 and self.args.decay_epoch>0):
                    decay_epoch = self.args.decay_epoch
                    lr = tf.train.exponential_decay(self.args.learn_rate, 
                                  global_step, 
                                  decay_epoch * self.args.batch_size, 
                                   self.args.decay_lr, staircase=True)
                else:
                    lr = self.args.learn_rate

                with tf.name_scope('optimizer'):
                    if(self.args.opt=='SGD'):
                        self.opt = tf.train.GradientDescentOptimizer(
                            learning_rate=lr)
                    elif(self.args.opt=='Adam'):
                        self.opt = tf.train.AdamOptimizer(
                                        learning_rate=lr)
                    elif(self.args.opt=='Adadelta'):
                        self.opt = tf.train.AdadeltaOptimizer(
                                        learning_rate=lr, 
                                        rho=0.9)
                    elif(self.args.opt=='Adagrad'):
                        self.opt = tf.train.AdagradOptimizer(
                                        learning_rate=lr)
                    elif(self.args.opt=='RMS'):
                        self.opt = tf.train.RMSPropOptimizer(
                                    learning_rate=lr)
                    elif(self.args.opt=='Moment'):
                        self.opt = tf.train.MomentumOptimizer(lr, 0.9)
                    elif(self.args.opt=='Adamax'):
                        self.opt = AdamaxOptimizer(lr)

                    # Use SGD at the end for better local minima
                    tvars = tf.trainable_variables()
                    def _none_to_zero(grads, var_list):
                        return [grad if grad is not None else tf.zeros_like(var)
                              for var, grad in zip(var_list, grads)]
                    if(self.args.clip_norm>0):
                        grads, _ = tf.clip_by_global_norm(
                                        tf.gradients(self.cost, tvars), 
                                        self.args.clip_norm)
                        with tf.name_scope('gradients'):
                            gradients = self.opt.compute_gradients(self.cost)
                            # Gradient Conversion
                            gradients = [(H2E_ball(grad),var) \
                                             for grad, var in gradients]
                            def ClipIfNotNone(grad):
                                if grad is None:
                                    return grad
                                grad = tf.clip_by_value(grad, -10, 10, 
                                                        name=None)
                                return tf.clip_by_norm(grad, 
                                                self.args.clip_norm)
                            if(self.args.clip_norm>0):
                                clip_g = [(ClipIfNotNone(grad), var) \
                                                for grad, var in gradients]
                            else:
                                clip_g = [(grad,var) for grad,var in gradients]
                      
                        with tf.control_dependencies(control_deps):
                            self.train_op = self.opt.apply_gradients(clip_g, 
                                                global_step=global_step)
                            self.wiggle_op = self.opt2.apply_gradients(clip_g, 
                                                global_step=global_step)
                   
                self.grads = _none_to_zero(tf.gradients(self.cost,tvars), tvars)
                self.merged_summary_op = tf.summary.merge_all(
                                        key=tf.GraphKeys.SUMMARIES)


                self.predict_op = self.output_pos
               
  







