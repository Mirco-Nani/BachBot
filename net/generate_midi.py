from __future__ import print_function, division

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
        
    def update(self):
        result = False
        new_actual = self.actual + 1
        new_percent_actual = 100*new_actual/self.total
        if self.percent_actual <= self.next_log <= new_percent_actual:
            print(str(new_actual)+"/"+str(self.total)+" - "+str(new_percent_actual)+"%")
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
    
    """
    encoder_input = [start_track]
    for i in range(encoder_size-1):
        r = random.randint(0,vocabulary_size-1)
        if r in not_allowed:
            r = wait_a_beat
        encoder_input.append(r)
    """
    encoder_input = song[0:encoder_size]
    
    """
    r = random.randint(0,vocabulary_size-1)
    if r in not_allowed:
        r = wait_a_beat
    
    first_decoder_input = [r] + [0 for i in range(decoder_size-1)]
    """
    first_decoder_input = [song[encoder_size]] + [0 for i in range(decoder_size-1)]
    
    song_begin = encoder_input + [first_decoder_input[0]]
    
    return np.array([encoder_input]), np.array([first_decoder_input]), song_begin
        
def flatten(arr):
    res = []
    for a in arr:
        res.append(a[0])
    return res

songs_destination = "../../experiments/midi/bach/exp02/generated/01/"
ensure_dir_exists(songs_destination)
        
params = load_json("../../experiments/midi/bach/exp02/config01.json", verbose=True)
params["feed_previous"] = True
#params["use_gpu"] = False


dataset_path = "/notebooks/recurrent-nets/workspace/datasets/midi/bach_flattened_encodings/default.json"
json_dataset = load_json(dataset_path, verbose=True)
json_dataset_song = json_dataset["encodings"]["jesu1.mid"]
midi_dataset = MidiDataset(json_dataset)
#encoder_input, first_decoder_input, song_begin = generate_song_sample(params["encoder_size"], params["decoder_size"], midi_dataset, json_dataset_song)
encoder_input, first_decoder_input, song_begin = generate_random_sample(params["encoder_size"], params["decoder_size"], midi_dataset)

vocabulary = midi_dataset.get_encoder().get_vocabulary()
end_track_message = vocabulary.index("END_TRACK")

for check in [5000]:#reversed(range(5000,10001,250)):
    
    #checkpoint= 10000 - check
    checkpoint = check

    print("CHECKPOINT:",checkpoint)
    params["start_checkpoint"] = checkpoint

    print("net instantiation...")
    net = ComplexRnn(params)
    print("building graph...")
    net.build_graph(verbose=True)
    print("initializing...")
    net.initialize(step=params["start_checkpoint"])

    print("net ready")

    #song_limit = 100000
    song_limit = 128

    song = [x for x in song_begin]
    next_enc_input = np.copy(encoder_input)
    next_dec_input = np.copy(first_decoder_input)

    found_end_track = False

    print("generating...")

    progress = ProgressLogger(song_limit,10)
    for i in range(song_limit):

        last_song_message = song[len(song)-1]

        pred = net.predict(next_enc_input, next_dec_input)
        flat_pred = flatten(pred)


        song.extend(flat_pred)

        if end_track_message in flat_pred:
            print("step",i,": found END_TRACK (",end_track_message, ") in", flat_pred) 
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
    print("saving midi to",os.path.abspath(os.path.join(songs_destination, "bachbot_"+str(checkpoint)+".mid")))
    encoder.decode_to_midi(song).save(os.path.join(songs_destination, "bachbot_"+str(checkpoint)+".mid"))
    print("midi saved to",os.path.abspath(os.path.join(songs_destination, "bachbot_"+str(checkpoint)+".mid")))
    
print("DONE")
