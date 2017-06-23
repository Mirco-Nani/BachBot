from __future__ import print_function
from mido import MidiFile, MetaMessage, Message, MidiTrack
from midi_dataset import normalize_midi

from os import listdir
from os.path import isfile, join


class ProgressLogger:
    def __init__(self, total, percent_step):
        self.total = total
        self.percent_step = percent_step
        self.actual = 0
        self.percent_actual = 0
        self.next_log = self.percent_step

        def log(to_log):
            print(to_log)

        self.log = log

    def set_logger(self, logger):
        self.log = logger

    def update(self):
        result = False
        new_actual = self.actual + 1
        new_percent_actual = 100 * new_actual / self.total
        if self.percent_actual <= self.next_log <= new_percent_actual:
            self.log(str(new_actual) + "/" + str(self.total) + " - " + str(new_percent_actual) + "%")
            # print(str(new_actual)+"/"+str(self.total)+" - "+str(new_percent_actual)+"%")
            self.next_log += self.percent_step
            result = True
        self.actual = new_actual
        self.percent_actual = new_percent_actual
        return result


master_defaults = {
    "time_signature.numerator": 4,
    "time_signature.denominator": 2, # quattro quarti
    "time_signature.clocks_per_click": 24,
    "time_signature.notated_32nd_notes_per_beat": 8,
    "key_signature.key" : "C", # Do maggiore
    "set_tempo.tempo" : 500000, #120bpm
    "ticks_per_beat" : 16,
    "note.velocity": 64,
    "note.channel": 0,
}

def midi_to_timesteps(normalized_midi, zeros=0, ones=1):
    if len(normalized_midi.tracks) != 1:
        print("midi is not normalized! It has ",len(normalized_midi.tracks), "tracks!")
        raise

    o = zeros
    l = ones

    state_vectors = {
        "NOTE": [o, o],
        "START_TRACK": [l, o],
        "END_TRACK": [o, l]
    }

    silence = [o for i in range(128)] + state_vectors["NOTE"]
    start_track = [o for i in range(128)] + state_vectors["START_TRACK"]
    end_track = [o for i in range(128)] + state_vectors["END_TRACK"]

    track = normalized_midi.tracks[0]
    pending_messages = []
    active_notes = list(silence)
    wait = 0


    timesteps = []


    for msg in track:
        """
        if msg.type in ["time_signature", "key_signature", "set_tempo"]:
            if len(timesteps)==0 :
                timesteps.append(start_track)
        """
        if msg.type in ["note_on", "note_off"]:
            if msg.time != 0:
                for i in range(msg.time):
                    timesteps.append(list(active_notes))

            #if msg.time == 0:
            if msg.type == "note_on":
                note = msg.note
                active_notes[note] = l
            elif msg.type == "note_off":
                note = msg.note
                active_notes[note] = o

            timesteps.append(list(active_notes))

    timesteps[0] = timesteps[0][:128] +  state_vectors["START_TRACK"]
    timesteps[len(timesteps)-1] = timesteps[len(timesteps)-1][:128] + state_vectors["END_TRACK"]
    #timesteps.append(end_track)
    return timesteps

"""
def timestep_diff(timestep1,timestep2):
    res = []
    same = True
    for i,t1 in enumerate(timestep1):
        diff = timestep1[i]-timestep2[i]
        res.append(diff)
        if diff != 0:
            same=False
    return res, same
"""

def timesteps_to_midi(timesteps, zeros=0, ones=1, defaults=master_defaults):
    o = zeros
    l = ones

    messages = []

    #HEADER
    time_signature = MetaMessage("time_signature")
    time_signature.numerator = defaults["time_signature.numerator"]
    time_signature.denominator = defaults["time_signature.denominator"]
    time_signature.clocks_per_click = defaults["time_signature.clocks_per_click"]
    time_signature.notated_32nd_notes_per_beat = defaults["time_signature.notated_32nd_notes_per_beat"]

    key_signature = MetaMessage("key_signature")
    key_signature.key = defaults["key_signature.key"]

    set_tempo = MetaMessage("set_tempo")
    set_tempo.tempo = defaults["set_tempo.tempo"]

    messages.extend([time_signature, key_signature, set_tempo])

    state_vectors = {
        "NOTE": [o, o],
        "START_TRACK": [l, o],
        "END_TRACK": [o, l]
    }

    silence = [o for i in range(128)] + state_vectors["NOTE"]

    wait = 0

    last_timestep = silence

    def timestep_diffs(timestep1, timestep2):
        res = []
        for i, t1 in enumerate(timestep1):
            diff = timestep1[i] - timestep2[i]
            if diff != 0:
                res.append(i)
        return res


    for timestep in timesteps:
        diffs = timestep_diffs(timestep[:128], last_timestep[:128])
        if(len(diffs)==0):
            wait+=1
        else:
            for d in diffs:
                if last_timestep[d] == o and timestep[d] == l:
                    message = Message("note_on", note=d, time=wait, channel=defaults["note.channel"], velocity=defaults["note.velocity"])
                    messages.append(message)
                    wait=0
                elif last_timestep[d] == l and timestep[d] == o:
                    message = Message("note_off", note=d, time=wait, channel=defaults["note.channel"])
                    messages.append(message)
                    wait = 0


        last_timestep = timestep

    messages.append(MetaMessage('end_of_track', time=wait))
    wait=0

    mid = MidiFile(ticks_per_beat=defaults["ticks_per_beat"])

    mid.tracks.append(MidiTrack(messages))

    return mid





mid = MidiFile('files/mir1.mid');
normalized_mid = normalize_midi(mid)
timesteps = midi_to_timesteps(normalized_mid, zeros=0, ones=1)

progress = ProgressLogger(len(timesteps),5)
with open("test.txt", "w") as out_file:
    for timestep in timesteps:
        for t in timestep:
            out_file.write("#" if t == 1 else " ")
        out_file.write("\n")
        progress.update()

mid = timesteps_to_midi(timesteps)

mid.save("files/test_single_track.mid")