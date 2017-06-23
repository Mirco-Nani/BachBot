from __future__ import print_function
from mido import MidiFile, Message, MetaMessage, MidiTrack, merge_tracks
import fractions

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
            print("Track", i);

        now = 0
        for message in track:
            if verbose:
                print(message)

            now += message.time
            absolute_messages.append({
                "msg" : message,
                "abs_time" : now
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


def add_defaults(single_track):
    time_signature = MetaMessage("time_signature")
    time_signature.numerator = 4
    time_signature.denominator = 2 #quattro quarti
    time_signature.clocks_per_click = 24
    time_signature.notated_32nd_notes_per_beat = 8

    key_signature = MetaMessage("key_signature")
    key_signature.key = 'C'  # C major

    set_tempo = MetaMessage("set_tempo")
    set_tempo.tempo = 400000 #500000  # 120bpm

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
        print("GCD:",times_gcd)

    return MidiTrack([msg.copy(time=msg.time / times_gcd) for msg in single_track]), times_gcd

def fixed_velocity(single_track, velocity=64):
    return MidiTrack([
         msg.copy(velocity=velocity) if msg.type in ["note_on", "note_off"] else msg.copy() for msg in single_track
     ])

def flatten_channels(single_track):
    return MidiTrack([
         msg.copy(channel=0) if msg.type in ["note_on", "note_off"] else msg.copy() for msg in single_track
     ])

def force_ticks_per_beat(single_track, ticks_per_beat, actual_ticks_per_beat):
    factor = float(actual_ticks_per_beat)/float(ticks_per_beat)
    return MidiTrack([msg.copy(time=int( float(msg.time) / factor )) for msg in single_track])

def print_track(track):
    for message in track:
        print(message)


def convert_midi(mid):
    single_track = new_merge_tracks(mid.tracks)
    single_track = fix_note_off(single_track)
    single_track = add_defaults(remove_meta(single_track))
    single_track, times_gcd = normalize_times(single_track)
    single_track = fixed_velocity(single_track)
    single_track = flatten_channels(single_track)

    ticks_per_beat = mid.ticks_per_beat / times_gcd
    forced_ticks_per_beat = 16
    forced_single_track = force_ticks_per_beat(single_track, forced_ticks_per_beat, ticks_per_beat)

    #new_mid = MidiFile(ticks_per_beat=ticks_per_beat)
    #new_mid.tracks.append(single_track)
    new_mid = MidiFile(ticks_per_beat=forced_ticks_per_beat)
    new_mid.tracks.append(forced_single_track)
    return new_mid

#def print

from os import listdir
from os.path import isfile, join


source = "files/bach/flattened"

"""
print("Collecting Files...")
midifiles = [MidiFile(join(source, f)) for f in listdir(source) if isfile(join(source, f))]
print("Converting Files...")
converted_midifiles = [convert_midi(f) for f in midifiles]
"""

"""
print("collecting and converting...")
converted_midifiles = [(convert_midi(MidiFile(join(source, f))), f) for f in listdir(source) if isfile(join(source, f))]

print("Collecting statistics..")

max_time = 0
min_note = 65535
max_note = 0
for mid,f in converted_midifiles:
    track = mid.tracks[0]
    for msg in track:
        if max_time < msg.time:
            max_time = msg.time
        if msg.type in ["note_on", "note_off"]:
            if msg.note < min_note:
                min_note = msg.note
            if msg.note > max_note:
                max_note = msg.note


print("max_time:", max_time)
print("min_note:", min_note)
print("max_note:", max_note)

ticks = {}
for mid,f in converted_midifiles:
    t = str(mid.ticks_per_beat)
    if t not in ticks:
        ticks[t] = [f]
    else:
        ticks[t].append(f)

print(ticks.keys())
print(ticks)
"""