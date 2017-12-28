#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import sys, os
import logging
sys.path.append("../")

from clustering.twopoisson_em_with_covariates import TwoPoissonEM

import tensorflow as tf
import tempfile



class TwoPoissonEM_NN(TwoPoissonEM):
    """
    Clustering of Poisson processes with 
    priors (pi) being a function (modeled by neural network) of covariates. 
    """    
    
    def __init__(self, users0, users1, features):
        super().__init__(users0, users1, features)
        
        input_dimension = features.shape[1]
        self.output_activations, self.loss, self.updates, self.data_batch, self.label_batch = self._build_graph(input_dimension)
        
        self.model_path = tempfile.mktemp(prefix="twopoisson-em-nn-model-pid%i-" % os.getpid())
                        
                        
    def _build_graph(self, input_dimension):
        tf.reset_default_graph()
    
        #input_dimension = train.shape[1]
        output_dimension = 1
        hidden1_units = 12
    
        data_batch =  tf.placeholder("float", shape=[None, input_dimension], name="data")
        label_batch = tf.placeholder("float", shape=[None, output_dimension], name="labels")
    
        weights_1 = tf.Variable(tf.truncated_normal([input_dimension, hidden1_units], stddev=1.0 / np.sqrt(float(input_dimension))),name='weights_1')
        #weights_1 = tf.get_variable("weights_1", [input_dimension, hidden1_units])
        biases_1 = tf.Variable(tf.truncated_normal([hidden1_units], stddev=1.0 / np.sqrt(float(hidden1_units))),name='biases_1')
        #biases_1 = tf.get_variable("biases_1", [hidden1_units])
        weights_2 = tf.Variable(tf.truncated_normal([hidden1_units, output_dimension], stddev=1.0 / np.sqrt(float(hidden1_units))),name='weights_2')
        #weights_2 = tf.get_variable("weights_2", [hidden1_units, output_dimension])
        biases_2 = tf.Variable(tf.truncated_normal([output_dimension], stddev=1.0 / np.sqrt(float(hidden1_units))),name='biases_2')
        #biases_2 = tf.get_variable("biases_2", [output_dimension])
    
        wx_b = tf.add(tf.matmul(data_batch, weights_1), biases_1)
        #hidden_activations = tf.nn.relu(wx_b)
        hidden_activations = tf.nn.sigmoid(wx_b)
        output_activations = tf.nn.sigmoid(tf.matmul(hidden_activations, weights_2)+biases_2, name="output_activations")
    
        loss = tf.nn.l2_loss(label_batch - output_activations)
        updates = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)
        #updates = tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.0001).minimize(loss)        
        return output_activations, loss, updates, data_batch, label_batch
    
    def train_graph(self, train, train_cls):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)    
            
            #restore session to continue training
            #saver = tf.train.import_meta_graph('/tmp/model-0.meta')    
            #saver = tf.train.Saver()
            #saver.restore(sess, '/tmp/model-0')
                        
            for i in range(2000):
                batch_loss, _, out = sess.run(
                    [self.loss, self.updates, self.output_activations], 
                    feed_dict = {
                        self.data_batch : train,
                        self.label_batch : np.matrix(train_cls).T
                    })
            
            logging.debug("[twopoisson_em][#E=%s,#M=%s][M][maximize_pi] saving model to %s" % 
                      (self.e_count, self.m_count, self.model_path))            
            tf.train.Saver().save(sess, self.model_path, global_step=0)
        return out[:,0]
            
        
    def predict(self, features):
        tf.reset_default_graph()        
        saver = tf.train.import_meta_graph(self.model_path+'-0.meta') 
        
        with tf.Session() as sess:    
            saver.restore(sess, self.model_path+"-0")
            graph = tf.get_default_graph()

            output_activations = graph.get_tensor_by_name("output_activations:0")
            data_batch = graph.get_tensor_by_name("data:0") 
            
            #init = tf.global_variables_initializer()
            #sess.run(init)    
            predictions = sess.run(
                            output_activations, 
                            feed_dict = {
                                data_batch : features,
                                #label_batch : test_labels
                            })                
        return predictions[:,0]
                        
        
    def maximize_pi(self):       
        matched_pi = self.train_graph(self.features, self.z[:,1])        
        self.pi = np.zeros( (len(self.features), 2) )
        self.pi[:,1] = matched_pi
        self.pi[:,0] = 1.0-self.pi[:,1]
           
           
           
           