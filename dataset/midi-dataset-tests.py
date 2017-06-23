from __future__ import print_function
from mido import MidiFile
from midi_dataset import MidiEncoder, print_midi

from os import listdir
from os.path import isfile, join

#ticks_per_beat = 16

defaults = {
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

encoder = MidiEncoder(defaults=defaults)

source = "files/bach/flattened"
destination = "files/bach/normalized"


midis = [(f,join(source, f)) for f in listdir(source) if isfile(join(source, f))]
for filename,path in midis:
    mid = MidiFile(path)
    print("normalizing",filename)
    normalized_mid = encoder.normalize_midi(mid)
    encodings = encoder.encode_midi(normalized_mid)
    decoded_mid = encoder.decode_to_midi(encodings)
    decoded_mid.save(join(destination,filename))

"""
mid = MidiFile('files/mir1.mid')
normalized_mid = encoder.normalize_midi(mid)
print(normalized_mid.ticks_per_beat)
print_midi(normalized_mid)

#verbose_encodings = encoder.encode_midi(normalized_mid, verbose_encoding=True)
encodings = encoder.encode_midi(normalized_mid)
verbose_encodings = encoder.decode_to_verbose(encodings)

print(verbose_encodings)
print(encodings)
print(len(encodings))

#decoded_mid = encoder.decode_to_midi(encodings[:64]+[4098])
decoded_mid = encoder.decode_to_midi(encodings)
decoded_mid.save("files/test_single_track.mid")
"""