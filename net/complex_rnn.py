from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import time
import os
import shutil

def ensure_dir_exists(directory):
    if not os.path.isdir(directory):
        directory = os.path.dirname(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def complex_rnn_master_defaults():
    return {
        # Parameters
        "learning_rate" : 0.001,
        "training_iters" : 30000,
        "batch_size" : 128,
        "display_step" : 10,
        "n_input_symbols" : 20,
        "n_output_symbols" : 20,
        "embedding_size" :4,
        "use_gpu" : True,
        "full_gpu": True,

        "save_step" : 20, # WARNING! saving takes a LOOOOOOT of time!
        "save_location" : "../experiments/toy_problem/checkpoints/",
        "tensorboard_location" : "../experiments/toy_problem/tensorboard/",
        "start_checkpoint" : 200,

        # Network Parameters
        "dropout_input_keep_prob" : 1.0,
        "dropout_output_keep_prob" : 1.0,
        "feed_previous" : False,
        "layers" : 1, # more layers means slower training time (obviously)
        "attention" : False, # attention makes training veeery slow 

        "encoder_size" : 10,
        "decoder_size" : 10,
        "n_hidden" : 64, # hidden layer num of features
        #"n_classes" : sequence_length, # (0 to sequence_length-1 digits)
        "n_classes" : 20
}



class ComplexRnn():
    def __init__(self, defaults):
        print("INITIALIZING NEW RNN")
        
        self.learning_rate = defaults["learning_rate"]
        #self.training_iters = defaults["training_iters"]
        self.batch_size = defaults["batch_size"]
        self.display_step = defaults["display_step"]
        self.n_input_symbols = defaults["n_input_symbols"]
        self.n_output_symbols = defaults["n_output_symbols"]
        
        if "embedding_size" in defaults:
            self.embedding_size = defaults["embedding_size"]
        else:
            self.embedding_size = defaults["n_classes"]
            
        
        self.use_gpu = defaults["use_gpu"]
        self.full_gpu = defaults["full_gpu"]
        self.save_step = defaults["save_step"]
        self.save_location = defaults["save_location"]
        if self.save_location is not None:
            ensure_dir_exists(self.save_location)
        
        self.start_checkpoint = defaults["start_checkpoint"]
        self.dropout_input_keep_prob = defaults["dropout_input_keep_prob"]
        self.dropout_output_keep_prob = defaults["dropout_output_keep_prob"]
        self.feed_previous = defaults["feed_previous"]
        self.layers = defaults["layers"]
        self.attention = defaults["attention"]
        self.encoder_size = defaults["encoder_size"]
        self.decoder_size = defaults["decoder_size"]
        self.n_hidden = defaults["n_hidden"]
        self.n_classes = defaults["n_classes"]
        if "tensorboard_location" in defaults:
            self.tensorboard_location = defaults["tensorboard_location"]
            #print(self.tensorboard_location)
        else:
            self.tensorboard_location = None
        
        self.session = None
        
        self.step = 0
        
        self.saver = None
        
        def log(to_log):
            print(to_log)
        
        self.log = log
        
        #self.build_graph()
        
    def set_logger(self, logger):
        self.log=logger
        
    def build_graph(self, verbose=False):
        tf.reset_default_graph()
        
        device = "/gpu:0" if self.use_gpu else "/cpu:0"
        
        current_device="/cpu:0" if not self.full_gpu else device
        with tf.device(current_device):
            if verbose:
                self.log("input manipulation on device: "+str(current_device))
                #print("input manipulation on device:",current_device)
        
            # tf Graph input
            x_encoder = tf.placeholder(tf.int32, [None, self.encoder_size], name="x_encoder")
            x_decoder = tf.placeholder(tf.int32, [None, self.decoder_size], name="x_decoder")
            expected_output = tf.placeholder(tf.int32, [None, self.decoder_size], name="expected_output")

            # dropout
            dropout_input_keep_prob_tensor = None
            dropout_output_keep_prob_tensor = None
            if self.dropout_input_keep_prob < 1.0 or self.dropout_output_keep_prob < 1.0:
                dropout_input_keep_prob_tensor = tf.placeholder(tf.float32, name="dropout_input_keep_prob_tensor")
                dropout_output_keep_prob_tensor = tf.placeholder(tf.float32, name="dropout_output_keep_prob_tensor")
                self.dropout_input_keep_prob_tensor = dropout_input_keep_prob_tensor
                self.dropout_output_keep_prob_tensor = dropout_output_keep_prob_tensor


            # GRAPH CREATION BEGIN
            if self.layers < 1:
                self.log("layers must be at least 1")
                #print("layers must be at least 1")
                raise

            #dropout lstm
            def build_simple_lstm():
                return tf.nn.rnn_cell.LSTMCell(self.n_hidden)

            def build_dropout_lstm():
                return tf.nn.rnn_cell.DropoutWrapper(
                            tf.nn.rnn_cell.LSTMCell(self.n_hidden),
                            dropout_input_keep_prob_tensor,
                            dropout_output_keep_prob_tensor
                        )

            if dropout_input_keep_prob_tensor is None and dropout_output_keep_prob_tensor is None:
                build_lstm = build_simple_lstm
            else:
                build_lstm = build_dropout_lstm


            # input conversions
            # We use dense embedding representation instead of one-hots

            # Permuting batch_size and n_steps: shape=(encoder_size, batch_size)
            x_inner_encoder = tf.transpose(x_encoder, [1, 0])
            # Reshaping to (encoder_size*batch_size)
            x_inner_encoder = tf.reshape(x_inner_encoder, [-1])
            # Split to get a list of 'encoder_size' tensors of shape (batch_size,)
            x_inner_encoder = tf.split(0, self.encoder_size, x_inner_encoder)
            if verbose:
                self.log("x_inner_encoder "+str(len(x_inner_encoder))+" "+str(x_inner_encoder[0].get_shape()))
                #print("x_inner_encoder",len(x_inner_encoder),x_inner_encoder[0].get_shape())

            # Permuting batch_size and n_steps: shape=(decoder_size, batch_size)
            x_inner_decoder = tf.transpose(x_decoder, [1, 0])
            # Reshaping to (decoder_size*batch_size)
            x_inner_decoder = tf.reshape(x_inner_decoder, [-1])
            # Split to get a list of 'decoder_size' tensors of shape (batch_size,)
            x_inner_decoder = tf.split(0, self.decoder_size, x_inner_decoder)
            if verbose:
                self.log("x_inner_decoder "+str(len(x_inner_decoder))+" "+str(x_inner_decoder[0].get_shape()))
                #print("x_inner_decoder",len(x_inner_decoder),x_inner_decoder[0].get_shape())


            # Permuting batch_size and n_steps: shape=(decoder_size, batch_size)
            target = tf.transpose(expected_output, [1, 0])
            # Reshaping to (decoder_size*batch_size)
            target = tf.reshape(target, [-1])
            # Split to get a list of 'decoder_size' tensors of shape (batch_size,)
            target = tf.split(0, self.decoder_size, target)
            if verbose:
                self.log("target "+str(len(target))+" "+str(target[0].get_shape()))
                #print("target",len(target),target[0].get_shape())


            # converting output to one-hot representation just to track accuracies: shape=(batch_size, decoder_size, n_output_symbols)

            #one_hot_y = tf.one_hot(expected_output, n_output_symbols, on_value=1.0, off_value=0.0, axis=-1, dtype=tf.float32, name=None)
            one_hot_y = tf.one_hot(expected_output, self.n_classes, on_value=1.0, off_value=0.0, axis=-1, dtype=tf.float32, name=None)

            # Permuting batch_size and n_steps: shape=(decoder_size, batch_size, n_output_symbols)
            one_hot_y = tf.transpose(one_hot_y, [1, 0, 2])
            # Reshaping to (decoder_size*batch_size, n_output_symbols)
            one_hot_y = tf.reshape(one_hot_y, [-1, self.n_classes])
            # Split to get a list of 'decoder_size' tensors of shape (batch_size, n_output_symbols)
            one_hot_y = tf.split(0, self.decoder_size, one_hot_y)
            if verbose:
                self.log("one_hot_y "+str(len(one_hot_y))+" "+str(one_hot_y[0].get_shape()))
                #print("one_hot_y",len(one_hot_y),one_hot_y[0].get_shape())

        with tf.device(device):
            if verbose:
                self.log("recurrent cells on device: "+str(device))
                #print("recurrent cells on device:",device)
            if self.layers == 1:
                lstm_cell = build_lstm()#tf.nn.rnn_cell.LSTMCell(n_hidden)#
            else:
                lstm_cells = [build_lstm() for i in range(self.layers)]#[tf.nn.rnn_cell.LSTMCell(n_hidden) for i in range(layers)]#
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        current_device="/cpu:0" if not self.full_gpu else device
        with tf.device(current_device):
            if verbose:
                self.log("embeddings on device: "+str(current_device))
                #print("embeddings on device:",current_device)
            
            
            
            if self.attention:
                outputs, states = tf.nn.seq2seq.embedding_attention_seq2seq(x_inner_encoder, x_inner_decoder, lstm_cell, 
                                                        self.n_input_symbols, self.n_output_symbols, 
                                                        self.embedding_size, feed_previous=self.feed_previous)
            else:
                outputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(x_inner_encoder, x_inner_decoder, lstm_cell, 
                                                        self.n_input_symbols, self.n_output_symbols, 
                                                        self.embedding_size, feed_previous=self.feed_previous)

            if verbose:
                self.log("outputs "+ str(len(outputs)) +" "+ str(outputs[0].get_shape()) )
                #print("outputs", len(outputs), outputs[0].get_shape())
                
        
            weights = [tf.ones([self.batch_size], dtype=tf.float32) for i in range(0,len(outputs))]
            
        current_device=device
        with tf.device(current_device):
            if verbose:
                self.log("loss and optimizer on device: "+str(current_device))
                #print("loss and optimizer on device:",current_device)
            
            loss = tf.nn.seq2seq.sequence_loss(outputs, target, weights)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            
        current_device="/cpu:0" if not self.full_gpu else device
        with tf.device(current_device):
            if verbose:
                self.log("model evaluation on device: "+str(current_device))
                #print("model evaluation on device:",current_device)

            perplexity = tf.exp(loss, name="perplexity")

            #evaluate_model
            accuracies = []
            predictions = []

            for i,output in enumerate(outputs):
                prediction = tf.argmax(output,1)
                accuracy = tf.reduce_mean( tf.cast( tf.equal(prediction , tf.argmax(one_hot_y[i],1)), tf.float32 ) )
                predictions.append(prediction)
                accuracies.append(accuracy)
                
            avg_accuracy = tf.reduce_mean( tf.pack(accuracies) )

            if verbose:
                self.log("accuracies:"+ str(len(accuracies)) + str(accuracies[0].get_shape()))
                #print("accuracies", len(accuracies), accuracies[0].get_shape())
                
            self.x_encoder = x_encoder
            self.x_decoder = x_decoder
            self.expected_output = expected_output
            self.predictions = predictions
            self.accuracies = accuracies
            self.avg_accuracy = avg_accuracy
            self.perplexity = perplexity
            self.optimizer = optimizer
            self.saver = tf.train.Saver(max_to_keep = None)
            
            if self.tensorboard_location is not None:
                tf.summary.scalar('avg_accuracy', self.avg_accuracy)
                tf.summary.scalar('perplexity', self.perplexity)
                self.merged = tf.summary.merge_all()
            
        
        return x_encoder, x_decoder, expected_output, predictions, accuracies, perplexity, optimizer
    
    def get_session(self):
        if self.session is None:
            if self.use_gpu:
                self.session = tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True)#,gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
                )
            else:
                self.session = tf.Session()
            if self.tensorboard_location is not None:
                if os.path.exists(self.tensorboard_location):
                    shutil.rmtree(self.tensorboard_location)
                ensure_dir_exists(self.tensorboard_location)
                self.train_writer = tf.summary.FileWriter(self.tensorboard_location, self.session.graph)
            
        return self.session
    
    def initialize(self, source=None, step=None):
        sess = self.get_session()
        if step is None and self.start_checkpoint <= 0:
            sess.run(tf.global_variables_initializer())
            #self.step = 1
        else:
            if source is None:
                source = self.save_location
            if step is None:
                step = self.start_checkpoint
            source = os.path.join(source,str(step)+".ckpt")
            self.saver.restore(sess, source)
            self.step = step + 1
            
    def get_step(self):
        return self.step
                
    def save_chackpoint(self, verbose=False):
        sess = self.get_session()
        destination = os.path.join(self.save_location,str(self.step)+".ckpt")
        path = self.saver.save(sess, destination)
        if verbose:
            self.log("Model saved in file "+str(path))
            #print("Model saved in file",path)
    
    def train(self, encoder_input, decoder_input, expected_output, sess=None):
        if sess is None:
            sess = self.get_session()
        self.step += 1
        if self.dropout_input_keep_prob < 1.0 or self.dropout_output_keep_prob < 1.0:
            return sess.run(self.optimizer, feed_dict={
                self.x_encoder:encoder_input, 
                self.x_decoder:decoder_input, 
                self.expected_output: expected_output,
                self.dropout_input_keep_prob_tensor : self.dropout_input_keep_prob,
                self.dropout_output_keep_prob_tensor : self.dropout_output_keep_prob
            })
        else:
            return sess.run(self.optimizer, feed_dict={
                self.x_encoder:encoder_input, 
                self.x_decoder:decoder_input, 
                self.expected_output: expected_output
            })
            
    def test(self, encoder_input, decoder_input, expected_output, sess=None):
        if sess is None:
            sess = self.get_session()
            
        if self.tensorboard_location is None:
            fetches = [self.accuracies, self.avg_accuracy, self.perplexity]
        else:
            fetches = [self.accuracies, self.avg_accuracy, self.perplexity, self.merged]
            
        if self.dropout_input_keep_prob < 1.0 or self.dropout_output_keep_prob < 1.0:
            res = sess.run(fetches, feed_dict={
                self.x_encoder:encoder_input, 
                self.x_decoder:decoder_input, 
                self.expected_output: expected_output,
                self.dropout_input_keep_prob_tensor : 1.0,
                self.dropout_output_keep_prob_tensor : 1.0
            })
        else:
            res = sess.run(fetches, feed_dict={
                self.x_encoder:encoder_input, 
                self.x_decoder:decoder_input, 
                self.expected_output: expected_output,
            })
            
        if self.tensorboard_location is None:
            return res
        else:
            summary = res[len(res)-1]
            self.train_writer.add_summary(summary, self.step)
            self.train_writer.flush()
            return res[0:len(res)-1]
        
        
    def predict(self, encoder_input, decoder_input, sess=None):
        if sess is None:
            sess = self.get_session()
        if self.dropout_input_keep_prob < 1.0 or self.dropout_output_keep_prob < 1.0:
            return sess.run(self.predictions, feed_dict={
                self.x_encoder:encoder_input, 
                self.x_decoder:decoder_input,
                self.dropout_input_keep_prob_tensor : 1.0,
                self.dropout_output_keep_prob_tensor : 1.0
            })
        else:
            return sess.run(self.predictions, feed_dict={
                self.x_encoder:encoder_input, 
                self.x_decoder:decoder_input
            })
        
    def train_loop(self, generate_batch, training_iters):
        start_time = time.time()
        
        sess = self.get_session()

        # Keep training until reach max iterations
        while self.step * self.batch_size < training_iters:
            enc_input, dec_input, exp_output = generate_batch(
                self.encoder_size, self.decoder_size, self.batch_size, self.n_classes)

            net.train(enc_input, dec_input, exp_output, sess)

            if self.step % self.display_step == 0:
                    acc = net.test(enc_input, dec_input, exp_output, sess)
                    
                    self.log("step "+str(self.step)+" perplexity: "+str(acc[1])+" avg "+str(acc[1])+" accuracies "+str(acc[0]))
                    #print("step",self.step,"perplexity:",acc[1],"avg",acc[1],"accuracies",acc[0])
                    
            if self.save_step > 0:
                if self.step % self.save_step == 0:
                    net.save_chackpoint()

        self.log("Optimization Finished!")
        #print("Optimization Finished!")

        # Calculate accuracy for 256 test examples
        test_len = 256
        enc_input, dec_input, exp_output = generate_batch(
            self.encoder_size, self.decoder_size, self.batch_size, self.n_classes)

        self.log("Testing Accuracy: "+ str(net.test(enc_input, dec_input, exp_output, sess)))
        #print("Testing Accuracy:", net.test(enc_input, dec_input, exp_output, sess))

        self.log("--- training took %s seconds ---" % (time.time() - start_time))
        #print("--- training took %s seconds ---" % (time.time() - start_time))
        self.close_session()
        
    def close_session(self):
        if self.tensorboard_location is not None:
            self.train_writer.close()
        
        if self.session is not None:
            self.session.close()
        self.session=None
        
