from __future__ import print_function

from midi_dataset import MidiDataset
from complex_rnn import ComplexRnn
import json
import os
import time


def AntiLoopRnn(ComplexRnn):
    def build_graph(self, verbose=False):
        tf.reset_default_graph()
        
        device = "/gpu:0" if self.use_gpu else "/cpu:0"
        
        current_device="/cpu:0" if not self.full_gpu else device
        with tf.device(current_device):
            if verbose:
                print("input manipulation on device:",current_device)
        
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
                print("layers must be at least 1")
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
                print("x_inner_encoder",len(x_inner_encoder),x_inner_encoder[0].get_shape())

            # Permuting batch_size and n_steps: shape=(decoder_size, batch_size)
            x_inner_decoder = tf.transpose(x_decoder, [1, 0])
            # Reshaping to (decoder_size*batch_size)
            x_inner_decoder = tf.reshape(x_inner_decoder, [-1])
            # Split to get a list of 'decoder_size' tensors of shape (batch_size,)
            x_inner_decoder = tf.split(0, self.decoder_size, x_inner_decoder)
            if verbose:
                print("x_inner_decoder",len(x_inner_decoder),x_inner_decoder[0].get_shape())


            # Permuting batch_size and n_steps: shape=(decoder_size, batch_size)
            target = tf.transpose(expected_output, [1, 0])
            # Reshaping to (decoder_size*batch_size)
            target = tf.reshape(target, [-1])
            # Split to get a list of 'decoder_size' tensors of shape (batch_size,)
            target = tf.split(0, self.decoder_size, target)
            if verbose:
                print("target",len(target),target[0].get_shape())


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
                print("one_hot_y",len(one_hot_y),one_hot_y[0].get_shape())

        with tf.device(device):
            if verbose:
                print("recurrent cells on device:",device)
            if self.layers == 1:
                lstm_cell = build_lstm()#tf.nn.rnn_cell.LSTMCell(n_hidden)#
            else:
                lstm_cells = [build_lstm() for i in range(self.layers)]#[tf.nn.rnn_cell.LSTMCell(n_hidden) for i in range(layers)]#
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        current_device="/cpu:0" if not self.full_gpu else device
        with tf.device(current_device):
            if verbose:
                print("embeddings on device:",current_device)
            
            if self.attention:
                outputs, states = tf.nn.seq2seq.embedding_attention_seq2seq(x_inner_encoder, x_inner_decoder, lstm_cell, 
                                                        self.n_input_symbols, self.n_output_symbols, 
                                                        self.n_classes, feed_previous=self.feed_previous)
            else:
                outputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(x_inner_encoder, x_inner_decoder, lstm_cell, 
                                                        self.n_input_symbols, self.n_output_symbols, 
                                                        self.n_classes, feed_previous=self.feed_previous)

            if verbose:
                print("outputs", len(outputs), outputs[0].get_shape())
                
        
            weights = [tf.ones([self.batch_size], dtype=tf.float32) for i in range(0,len(outputs))]
            
        current_device=device
        with tf.device(current_device):
            ### BEGIN GRAPH EDITING ###
            
            
            if verbose:
                print("loss and optimizer on device:",current_device)
            
            loss = tf.nn.seq2seq.sequence_loss(outputs, target, weights)
            
            ### END GRAPH EDITING ###

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            
            
            
        current_device="/cpu:0" if not self.full_gpu else device
        with tf.device(current_device):
            if verbose:
                print("model evaluation on device:",current_device)

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
                print("accuracies", len(accuracies), accuracies[0].get_shape())
                
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


def ensure_dir_exists(directory):
    if not os.path.isdir(directory):
        directory = os.path.dirname(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_json(path, verbose=False):
    if os.path.isfile(path) : 
        if(verbose):
            print("opening"+" "+os.path.abspath(path))
        result = json.load(open( path , "r" ))
        if(verbose):
            print("json retreived succesfully")
        return result
    else:
        raise Exception(os.path.abspath(path)+" does not exists")
        
def save_json(dictionary, path, verbose=False):
    ensure_dir_exists(path)
    if(verbose):
        print('saving '+os.path.abspath(path)+" ...")
    json.dump(dictionary, open(path,'w'), sort_keys=True, indent=4)
    if(verbose):
        print(os.path.abspath(path)+" saved successfully")
        

dataset_path = "/notebooks/recurrent-nets/workspace/datasets/midi/bach_flattened_encodings/default.json"
json_dataset = load_json(dataset_path, verbose=True)
dataset = MidiDataset(json_dataset)

        
params = {
    "target" : "complex_rnn",
    
    # Parameters
    "learning_rate" : 0.001,
    "training_iters" : 10000*128,
    "batch_size" : 128,
    "display_step" : 10,
    "n_input_symbols" : dataset.get_vocabulary_size(),
    "n_output_symbols" : dataset.get_vocabulary_size(),
    "use_gpu" : True,
    "full_gpu": True,

    "save_step" : 250, # WARNING! saving takes a LOOOOOOT of time!
    "save_location" : "../../experiments/midi/bach/exp01/checkpoints/",
    "tensorboard_location" : "../../experiments/midi/bach/exp01/tensorboard/",
    "start_checkpoint" : 5000,

    # Network Parameters
    "dropout_input_keep_prob" : 1.0,
    "dropout_output_keep_prob" : 1.0,
    "feed_previous" : False,
    "layers" : 2, # more layers means slower training time (obviously)
    "attention" : False, # attention makes training veeery slow 

    "encoder_size" : 64,
    "decoder_size" : 64,
    "n_hidden" : 512, # hidden layer num of features
    #"n_classes" : sequence_length, # (0 to sequence_length-1 digits)
    "n_classes" : dataset.get_vocabulary_size()
}

save_json(params, "../../experiments/midi/bach/exp01/config01.json", verbose=True)

print("net instantiation...")
net = ComplexRnn(params)
print("building graph...")
net.build_graph(verbose=True)
print("initializing...")
net.initialize()



print("start training")

start_time = time.time()

# Keep training until reach max iterations
while net.get_step() * params["batch_size"] < params["training_iters"]:
    enc_input, dec_input, exp_output = dataset.generate_np_batch( #generate_batch(
        params["encoder_size"], params["decoder_size"], params["batch_size"])
    
    net.train(enc_input, dec_input, exp_output)
    
    if net.get_step() % params["display_step"] == 0:
            acc = net.test(enc_input, dec_input, exp_output)
            
            #print("step",net.get_step(),"perplexity:",acc[2],"avg",acc[1],"accuracies",acc[0])
            print("step",net.get_step(),"perplexity:",acc[2],"avg",acc[1])
    
    if params["save_step"] > 0:
        if net.get_step() % params["save_step"] == 0:
            net.save_chackpoint(verbose=True)

print("Optimization Finished!")

# Calculate accuracy for 256 test examples
test_len = 256
enc_input, dec_input, exp_output = dataset.generate_np_batch( #generate_batch(
    params["encoder_size"], params["decoder_size"], params["batch_size"])

print("Testing Accuracy:", net.test(enc_input, dec_input, exp_output))

print("--- training took %s seconds ---" % (time.time() - start_time))

net.close_session()