from __future__ import print_function

from midi_dataset import MidiDataset
from complex_rnn import ComplexRnn
import json
import os
import time
import random
import numpy as np

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
        
class ProgressLogger:
    def __init__(self,total,percent_step):
        self.total = total
        self.percent_step = percent_step
        self.actual = 0
        self.percent_actual = 0
        self.next_log = self.percent_step
        
        def log(to_log):
            print(to_log)
        
        self.log = log
        
    def set_logger(self, logger):
        self.log=logger
        
    def update(self):
        result = False
        new_actual = self.actual + 1
        new_percent_actual = 100*new_actual/self.total
        if self.percent_actual <= self.next_log <= new_percent_actual:
            self.log(str(new_actual)+"/"+str(self.total)+" - "+str(new_percent_actual)+"%")
            #print(str(new_actual)+"/"+str(self.total)+" - "+str(new_percent_actual)+"%")
            self.next_log += self.percent_step
            result = True
        self.actual = new_actual
        self.percent_actual = new_percent_actual
        return result

        
def generate_random_sample(encoder_size, decoder_size, midi_dataset):
    #encoder_input = np.array([[13, 14, 15, 16, 17, 18, 19, 0, 1, 2]])
    #first_decoder_input = np.array([[3,0,0,0,0,0,0,0,0,0]])
    
    vocabulary_size = midi_dataset.get_vocabulary_size()
    vocabulary = midi_dataset.get_encoder().get_vocabulary()
    
    start_track = vocabulary.index("START_TRACK")
    end_track = vocabulary.index("END_TRACK")
    wait_a_beat = vocabulary.index("WAIT_A_BEAT")
    unknown = vocabulary.index("UNKNOWN")
    
    not_allowed = [start_track, end_track, unknown]
    
    encoder_input = [start_track]
    for i in range(encoder_size-1):
        r = random.randint(0,vocabulary_size-1)
        if r in not_allowed:
            r = wait_a_beat
        encoder_input.append(r)
    
    r = random.randint(0,vocabulary_size-1)
    if r in not_allowed:
        r = wait_a_beat
    
    first_decoder_input = [r] + [0 for i in range(decoder_size-1)]
    
    song_begin = encoder_input + [first_decoder_input[0]]
    
    return np.array([encoder_input]), np.array([first_decoder_input]), song_begin

def generate_song_sample(encoder_size, decoder_size, midi_dataset, song):
    #encoder_input = np.array([[13, 14, 15, 16, 17, 18, 19, 0, 1, 2]])
    #first_decoder_input = np.array([[3,0,0,0,0,0,0,0,0,0]])
    
    vocabulary_size = midi_dataset.get_vocabulary_size()
    vocabulary = midi_dataset.get_encoder().get_vocabulary()
    
    start_track = vocabulary.index("START_TRACK")
    end_track = vocabulary.index("END_TRACK")
    wait_a_beat = vocabulary.index("WAIT_A_BEAT")
    unknown = vocabulary.index("UNKNOWN")
    
    not_allowed = [start_track, end_track, unknown]
    
    encoder_input = song[0:encoder_size]
    
    first_decoder_input = [song[encoder_size]] + [0 for i in range(decoder_size-1)]
    
    song_begin = encoder_input + [first_decoder_input[0]]
    
    return np.array([encoder_input]), np.array([first_decoder_input]), song_begin
        
def flatten(arr):
    res = []
    for a in arr:
        res.append(a[0])
    return res

dataset_path = "/notebooks/recurrent-nets/workspace/datasets/midi/bach_flattened_encodings/default.json"
json_dataset = load_json(dataset_path, verbose=True)
dataset = MidiDataset(json_dataset)
midi_dataset = dataset

"""
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
"""
experiments_path = "../../experiments/midi/bach/"

parameters = [
    ("exp06" , {
        "target" : "complex_rnn",

        # Parameters
        "learning_rate" : 0.001,
        "training_iters" : 5000*128,
        "batch_size" : 128,
        "display_step" : 10,
        "n_input_symbols" : dataset.get_vocabulary_size(),
        "n_output_symbols" : dataset.get_vocabulary_size(),
        "embedding_size" : 128,
        "use_gpu" : True,
        "full_gpu": True,

        "save_step" : 250, # WARNING! saving takes a LOOOOOOT of time!
        #"save_location" : os.path.join(experiment_path,"checkpoints/"),#"../../experiments/midi/bach/exp02/checkpoints/",
        #"tensorboard_location" : os.path.join(experiment_path,"tensorboard/"),#"../../experiments/midi/bach/exp02/tensorboard/",
        "start_checkpoint" : 2500,

        # Network Parameters
        "dropout_input_keep_prob" : 1.0,
        "dropout_output_keep_prob" : 1.0,
        "feed_previous" : False,
        "layers" : 2, # more layers means slower training time (obviously)
        "attention" : True, # attention makes training veeery slow (and net veery big)

        "encoder_size" : 64,
        "decoder_size" : 64,
        "n_hidden" : 512, # hidden layer num of features
        #"n_classes" : sequence_length, # (0 to sequence_length-1 digits)
        "n_classes" : dataset.get_vocabulary_size()
    })
    
]

#dataset_path = "/notebooks/recurrent-nets/workspace/datasets/midi/bach_flattened_encodings/default.json"
#json_dataset = load_json(dataset_path, verbose=True)
json_dataset_song = json_dataset["encodings"]["jesu1.mid"]
#midi_dataset = MidiDataset(json_dataset)
vocabulary = midi_dataset.get_encoder().get_vocabulary()
end_track_message = vocabulary.index("END_TRACK")

for i,experiment in enumerate(parameters):
    
    exp, params = parameters[i]
    
    print("Executing experiment:",exp,"INDEX:",i)
    #param=parameters[exp]
    experiment_path = os.path.join(experiments_path,exp)
    params["save_location"] = os.path.join(experiment_path,"checkpoints/")
    params["tensorboard_location"] = os.path.join(experiment_path,"tensorboard/")
    #params=parameters

    #save_json(params, "../../experiments/midi/bach/exp02/config01.json", verbose=True)
    save_json(params, os.path.join(experiment_path,"config02.json"), verbose=True)
    
    
    logs_destination = os.path.join(experiment_path,"logs02.out")
    out_file = open(logs_destination,"w")
    def logger(to_log):
        print(to_log)
        out_file.write(to_log+"\n")
        out_file.flush()
    

    ### TRAINING ###
    
    logger("net instantiation...")
    net = ComplexRnn(params)
    net.set_logger(logger)
    logger("building graph...")
    net.build_graph(verbose=True)
    logger("initializing...")
    net.initialize()



    logger("start training")

    start_time = time.time()

    # Keep training until reach max iterations
    while net.get_step() * params["batch_size"] < params["training_iters"]:
        enc_input, dec_input, exp_output = dataset.generate_np_batch( #generate_batch(
            params["encoder_size"], params["decoder_size"], params["batch_size"])

        net.train(enc_input, dec_input, exp_output)

        if net.get_step() % params["display_step"] == 0:
                acc = net.test(enc_input, dec_input, exp_output)

                #print("step",net.get_step(),"perplexity:",acc[2],"avg",acc[1],"accuracies",acc[0])
                logger("step "+str(net.get_step())+" perplexity: "+str(acc[2])+" avg "+str(acc[1]))

        if params["save_step"] > 0:
            if net.get_step() % params["save_step"] == 0:
                net.save_chackpoint(verbose=True)

    logger("Optimization Finished!")

    # Calculate accuracy for 256 test examples
    test_len = 256
    enc_input, dec_input, exp_output = dataset.generate_np_batch( #generate_batch(
        params["encoder_size"], params["decoder_size"], params["batch_size"])

    logger("Testing Accuracy: "+ str(net.test(enc_input, dec_input, exp_output)))

    logger("--- training took %s seconds ---" % (time.time() - start_time))

    net.close_session()
    
    ### END TRAINING ###
    
    ### START GENERATING ###
    
    songs_destination = os.path.join(experiment_path,"generated/default/")#"../../experiments/midi/bach/exp02/generated/01/"
    ensure_dir_exists(songs_destination)

    #params = load_json("../../experiments/midi/bach/exp02/config01.json", verbose=True)
    params["feed_previous"] = True
    
    
    #encoder_input, first_decoder_input, song_begin = generate_song_sample(params["encoder_size"], params["decoder_size"], midi_dataset, json_dataset_song)
    encoder_input, first_decoder_input, song_begin = generate_random_sample(params["encoder_size"], params["decoder_size"], midi_dataset)
    
    training_end = params["training_iters"]/params["batch_size"] +1
    for check in reversed(range(params["start_checkpoint"],training_end,params["save_step"])):
        if check == 0:
            continue
        
        checkpoint = check
        
        logger("CHECKPOINT: "+str(checkpoint))
        params["start_checkpoint"] = checkpoint

        logger("net instantiation...")
        net = ComplexRnn(params)
        net.set_logger(logger)
        logger("building graph...")
        net.build_graph(verbose=True)
        logger("initializing...")
        net.initialize(step=params["start_checkpoint"])

        logger("net ready")

        #song_limit = 100000
        song_limit = 128

        song = [x for x in song_begin]
        next_enc_input = np.copy(encoder_input)
        next_dec_input = np.copy(first_decoder_input)

        found_end_track = False

        logger("generating...")

        progress = ProgressLogger(song_limit,10)
        progress.set_logger(logger)
        for i in range(song_limit):

            last_song_message = song[len(song)-1]

            pred = net.predict(next_enc_input, next_dec_input)
            flat_pred = flatten(pred)


            song.extend(flat_pred)

            if end_track_message in flat_pred:
                logger("step "+str(i)+": found END_TRACK ("+str(end_track_message)+") in "+str( flat_pred )) 
                #print("step",i,": found END_TRACK (",end_track_message, ") in", flat_pred) 
                found_end_track = True
                break

            next_enc_input = [last_song_message] + flat_pred
            first_dec_input = next_enc_input.pop()
            next_dec_input = [first_dec_input] + [0 for i in range(params["decoder_size"]-1)]
            next_enc_input = np.array([next_enc_input])
            next_dec_input = np.array([next_dec_input])

            progress.update()

        if not found_end_track:
            song.append(end_track_message)

        save_json(song, os.path.join(songs_destination, "bachbot_"+str(checkpoint)+".json"), verbose=True)
        encoder = midi_dataset.get_encoder()
        logger("saving midi to "+os.path.abspath(os.path.join(songs_destination, "bachbot_"+str(checkpoint)+".mid")))
        encoder.decode_to_midi(song).save(os.path.join(songs_destination, "bachbot_"+str(checkpoint)+".mid"))
        logger("midi saved to "+os.path.abspath(os.path.join(songs_destination, "bachbot_"+str(checkpoint)+".mid")))
    
    
    ### END GENERATING ###
    out_file.close()
    