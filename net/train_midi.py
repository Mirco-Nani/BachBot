from __future__ import print_function

from midi_dataset import MidiDataset
from complex_rnn import ComplexRnn
import json
import os
import time

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

experiment_path = "../../experiments/midi/bach/exp02/";
        
params = {
    "target" : "complex_rnn",
    
    # Parameters
    "learning_rate" : 0.001,
    "training_iters" : 5000*128,
    "batch_size" : 128,
    "display_step" : 10,
    "n_input_symbols" : dataset.get_vocabulary_size(),
    "n_output_symbols" : dataset.get_vocabulary_size(),
    "use_gpu" : True,
    "full_gpu": True,

    "save_step" : 250, # WARNING! saving takes a LOOOOOOT of time!
    "save_location" : os.path.join(experiment_path,"checkpoints/"),#"../../experiments/midi/bach/exp02/checkpoints/",
    "tensorboard_location" : os.path.join(experiment_path,"tensorboard/"),#"../../experiments/midi/bach/exp02/tensorboard/",
    "start_checkpoint" : 0,

    # Network Parameters
    "dropout_input_keep_prob" : 1.0,
    "dropout_output_keep_prob" : 1.0,
    "feed_previous" : True,
    "layers" : 2, # more layers means slower training time (obviously)
    "attention" : False, # attention makes training veeery slow 

    "encoder_size" : 64,
    "decoder_size" : 64,
    "n_hidden" : 512, # hidden layer num of features
    #"n_classes" : sequence_length, # (0 to sequence_length-1 digits)
    "n_classes" : dataset.get_vocabulary_size()
}

#save_json(params, "../../experiments/midi/bach/exp02/config01.json", verbose=True)
save_json(params, os.path.join(experiment_path,"config01.json"), verbose=True)

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