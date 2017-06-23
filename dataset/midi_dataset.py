from __future__ import print_function
from mido import MidiFile, Message, MetaMessage, MidiTrack
import fractions


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

def gcd(numbers):
    if len(numbers) > 2:
        return reduce(lambda x, y: fractions.gcd(x, y), numbers)
    else:
        return fractions.gcd(numbers[0], numbers[1])

def save_track(track, path, ticks_per_beat = None):
    if ticks_per_beat:
        new_mid = MidiFile(ticks_per_beat = ticks_per_beat)
    else:
        new_mid = MidiFile()
    new_mid.tracks.append(track)
    print("new_mid.ticks_per_beat", new_mid.ticks_per_beat)
    new_mid.save(path)

def new_merge_tracks(tracks, verbose=False):
    absolute_messages = []
    for i,track in enumerate(tracks):
        if verbose:
            print("Track", i)

        now = 0
        for message in track:
            if verbose:
                print(message)

            now += message.time
            absolute_messages.append({
                "msg": message,
                "abs_time": now
            })

    absolute_messages.sort(key=lambda msg: msg["abs_time"])

    merged_messages = []

    now = 0
    for abs_msg in absolute_messages:
        new_msg = abs_msg["msg"].copy(time = (abs_msg["abs_time"] - now))
        merged_messages.append(new_msg)
        now = abs_msg["abs_time"]

    #return MidiTrack([msg["msg"] for msg in absolute_messages])
    return MidiTrack(merged_messages)

def build_note_off(channel,note,velocity,time):
    m = Message("note_off")
    m.channel = channel
    m.note = note
    m.velocity = velocity
    m.time = time
    return m

def fix_note_off(track):
    fixed_messages=[
        build_note_off(msg.channel,msg.note,msg.velocity, msg.time) if msg.type == "note_on" and msg.velocity == 0 else msg for msg in track
    ]
    return MidiTrack(fixed_messages)

def fix_end_of_track(track):
    fixed_messages = [
        msg for msg in track if msg.type != "end_of_track"
    ]
    fixed_messages.append(MetaMessage('end_of_track'))
    return MidiTrack(fixed_messages)

def remove_meta(single_track):
    return MidiTrack([
        message for message in single_track if not isinstance(message, MetaMessage) and message.type in ["note_on", "note_off"]
    ])


def add_defaults(single_track, defaults=master_defaults):
    time_signature = MetaMessage("time_signature")
    time_signature.numerator = defaults["time_signature.numerator"]
    time_signature.denominator = defaults["time_signature.denominator"]
    time_signature.clocks_per_click = defaults["time_signature.clocks_per_click"]
    time_signature.notated_32nd_notes_per_beat = defaults["time_signature.notated_32nd_notes_per_beat"]

    key_signature = MetaMessage("key_signature")
    key_signature.key = defaults["key_signature.key"]

    set_tempo = MetaMessage("set_tempo")
    set_tempo.tempo = defaults["set_tempo.tempo"]

    msgs = [time_signature, key_signature, set_tempo]
    msgs.extend(single_track)
    msgs.append(MetaMessage('end_of_track'))

    return MidiTrack(msgs)

def get_times(single_track):
    times = []
    for msg in single_track:
        if msg.time not in times and msg.time != 0:
            times.append(msg.time)
    return times

def normalize_times(single_track, verbose=False):
    times_gcd = gcd(get_times(single_track))
    if verbose:
        print("GCD:", times_gcd)

    return MidiTrack([msg.copy(time=msg.time / times_gcd) for msg in single_track]), times_gcd

def fixed_velocity(single_track, velocity=64):
    return MidiTrack([
         msg.copy(velocity=velocity) if msg.type in ["note_on", "note_off"] else msg.copy() for msg in single_track
     ])

def flatten_channels(single_track, channel=0):
    return MidiTrack([
         msg.copy(channel=channel) if msg.type in ["note_on", "note_off"] else msg.copy() for msg in single_track
     ])

def force_ticks_per_beat(single_track, ticks_per_beat, actual_ticks_per_beat):
    factor = float(actual_ticks_per_beat)/float(ticks_per_beat)
    return MidiTrack([msg.copy(time=int( float(msg.time) / factor )) for msg in single_track])

def print_track(track):
    for message in track:
        print(message)

def print_midi(midifile):
    for i,track in enumerate(midifile.tracks):
        print("Track", i)
        print_track(track)


def normalize_midi(mid, defaults=master_defaults):
    ticks_per_beat=master_defaults["ticks_per_beat"]

    single_track = new_merge_tracks(mid.tracks)
    single_track = fix_note_off(single_track)
    single_track = add_defaults(remove_meta(single_track), defaults=defaults)
    single_track, times_gcd = normalize_times(single_track)
    single_track = fixed_velocity(single_track, velocity=defaults["note.velocity"])
    single_track = flatten_channels(single_track, channel=defaults["note.channel"])

    resulting_ticks_per_beat = mid.ticks_per_beat / times_gcd
    forced_ticks_per_beat = ticks_per_beat
    forced_single_track = force_ticks_per_beat(single_track, forced_ticks_per_beat, resulting_ticks_per_beat)

    #new_mid = MidiFile(ticks_per_beat=ticks_per_beat)
    #new_mid.tracks.append(single_track)
    new_mid = MidiFile(ticks_per_beat=forced_ticks_per_beat)
    new_mid.tracks.append(forced_single_track)
    return new_mid


def generate_vocabulary(ticks_per_beat=master_defaults["ticks_per_beat"]):
    vocabulary = []

    # note_on and note_off messages
    for note_type in ["note_on", "note_off"]:
        for note in range(128):
            for time in range(ticks_per_beat):
                vocabulary.append(note_type + "-" + str(note) + "-" + str(time))

    # special_messages
    vocabulary.extend([
        "WAIT_A_BEAT",
        "START_TRACK",
        "END_TRACK",
        "UNKNOWN"
    ])

    encodings = dict( [ (v,i) for i,v in enumerate(vocabulary)] )
    decodings = dict( [ (i,v) for i,v in enumerate(vocabulary)] )

    return vocabulary, encodings, decodings

"""
def message_to_encoding(message, encodings, ticks_per_beat, verbose_encoding = False):

    def enc(enc_str):
        return encodings[enc_str] if not verbose_encoding else enc_str

    if message.type in ['time_signature', 'key_signature', 'set_tempo']:
        return [enc("START_TRACK")]
    if message.type == 'end_of_track':
        return [enc("END_TRACK")]

    if message.type in ["note_on", "note_off"]:
        result = []
        beats_to_wait = int( float(message.time)/float(ticks_per_beat) )
        for i in range(beats_to_wait):
            result.append(enc("WAIT_A_BEAT"))
        remaining_time = message.time % ticks_per_beat
        result.append(enc( message.type+"-"+message.note+"-"+remaining_time ))
        return result

    return [enc("UNKNOWN")]
"""

def message_to_encoding(message, ticks_per_beat):


    if message.type in ['time_signature', 'key_signature', 'set_tempo']:
        return ["START_TRACK"]
    if message.type == 'end_of_track':
        return ["END_TRACK"]

    if message.type in ["note_on", "note_off"]:
        result = []
        beats_to_wait = int( float(message.time)/float(ticks_per_beat) )
        for i in range(beats_to_wait):
            result.append("WAIT_A_BEAT")
        remaining_time = message.time % ticks_per_beat
        result.append( message.type+"-"+str(message.note)+"-"+str(remaining_time) )
        return result

    return ["UNKNOWN"]


def encode_midi(mid, encodings, ticks_per_beat = 16, verbose_encoding = False):
    if len(mid.tracks) != 1:
        print('midi must be single track, normalize it first with "normalize_midi" ')
        raise

    def enc(enc_str_list):
        return [encodings[s] for s in enc_str_list] if not verbose_encoding else enc_str_list


    track = mid.tracks[0]

    result = []

    for message in track:
        encoding_str = message_to_encoding(message, ticks_per_beat)

        #header messages at start of track
        if encoding_str[0] == "START_TRACK":
            if len(result) == 0:
                result.extend(enc(encoding_str))
        else:
            result.extend(enc(encoding_str))

    return result


def encoding_to_message(encoding, decodings, defaults=master_defaults):

    message = decodings[encoding]

    if message == "START_TRACK":
        #header defaults
        time_signature = MetaMessage("time_signature")
        time_signature.numerator = defaults["time_signature.numerator"]
        time_signature.denominator = defaults["time_signature.denominator"]
        time_signature.clocks_per_click = defaults["time_signature.clocks_per_click"]
        time_signature.notated_32nd_notes_per_beat = defaults["time_signature.notated_32nd_notes_per_beat"]

        key_signature = MetaMessage("key_signature")
        key_signature.key = defaults["key_signature.key"]

        set_tempo = MetaMessage("set_tempo")
        set_tempo.tempo = defaults["set_tempo.tempo"]

        return [time_signature, key_signature, set_tempo]

    if message == "END_TRACK":
        return [MetaMessage('end_of_track')]

    if message == "WAIT_A_BEAT":
        return [message]

    if message.startswith("note"):
        parts = message.split("-")
        res = Message(parts[0])
        res.note = int(parts[1])
        res.time = int(parts[2])
        # defaults
        res.channel = defaults["note.channel"]
        res.velocity = defaults["note.velocity"]
        return [res]

    return []

def decode_to_midi(encs, decodings, defaults=master_defaults):
    ticks_per_beat = defaults["ticks_per_beat"]
    mid = MidiFile(ticks_per_beat = ticks_per_beat)
    messages = []
    waiting_time = 0
    for enc in encs:
        message = encoding_to_message(enc, decodings, defaults=defaults)
        if isinstance(message[0], basestring):
            if message[0] == "UNKNOWN":
                continue
            elif message[0] == "WAIT_A_BEAT":
                waiting_time += ticks_per_beat
        else:
            if message[0].type in ["note_on", "note_off"]:
                message[0].time += waiting_time
                waiting_time = 0
                messages.extend(message)
            else:
                messages.extend(message)

    mid.tracks.append(MidiTrack(messages))
    return mid


class MidiEncoder():
    def __init__(self, defaults=master_defaults, ticks_per_beat=None):
        self.defaults = defaults

        if ticks_per_beat is not None:
            self.defaults["ticks_per_beat"]=ticks_per_beat

        self.ticks_per_beat = self.defaults["ticks_per_beat"]
        self.vocabulary, self.encodings, self.decodings = generate_vocabulary(ticks_per_beat=self.defaults["ticks_per_beat"])


    def normalize_midi(self, midifile):
        return normalize_midi(midifile, defaults=self.defaults)

    def encode_midi(self, midifile):
        return encode_midi(midifile, self.encodings, ticks_per_beat = self.ticks_per_beat)

    def decode_to_midi(self, encodings):
        return decode_to_midi(encodings, self.decodings, defaults=self.defaults)
        #return decode_to_midi(encodings, self.decodings, ticks_per_beat = self.ticks_per_beat)

    def decode_to_verbose(self, encodings):
        return [self.decodings[e] for e in encodings]

    def normalize_encodings(self, encodings):
        pad = int( float(self.vocabulary)/2.0 )
        return [e - pad for e in encodings]

    def denormalize_encodings(self, encodings):
        pad = int( float(self.vocabulary)/2.0 )
        return [e + pad for e in encodings]