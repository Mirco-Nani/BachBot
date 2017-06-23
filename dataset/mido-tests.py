from __future__ import print_function

from mido import MidiFile, Message, MetaMessage, MidiTrack, merge_tracks
import mido


def to_abstime(messages):
    """Convert messages to absolute time."""
    now = 0
    for msg in messages:
        now += msg.time
        yield msg.copy(time=now)

def new_merge_tracks(tracks):
    """Returns a MidiTrack object with all messages from all tracks.

    The messages are returned in playback order with delta times
    as if they were all in one track.
    """
    messages = []
    for track in tracks:
        messages.extend(to_abstime(track))

    messages.sort(key=lambda msg: msg.time)

    return MidiTrack(messages)




mid = MidiFile('files/mir1.mid');
#mid = MidiFile('files/bach/bach/chorales/01ausmei.mid');


for message in mid.play():
    print(message)

"""
new_mid = MidiFile()

print(len(mid.tracks))
for track in [new_merge_tracks(mid.tracks)]:
    new_track = MidiTrack()
    for msg in track:
        #print(msg.time)
        new_track.append(msg)
    new_mid.tracks.append(new_track)

new_mid.save('files/mir1_single_track.mid')
"""


"""
channels = []
notes = []
velocities = []
c=0
others = 0
for message in mid:
    if not isinstance(message, MetaMessage) and message.type in ["note_on", "note_off"]:
        #print(message)
        if message.channel not in channels:
            channels.append(message.channel)

        if message.note not in notes:
            notes.append(message.note)

        if message.velocity not in velocities:
            velocities.append(message.velocity)

        c+=1
        if c == 100:
            print(message)

        #track.append(message)
        track.append(Message(message.type, note=message.note, velocity=0, time=0))

    else:
        message.time=0
        track.append(message)
        others += 1

new_mid.save('files/mir1_notes_only.mid')

print("channels:", len(channels), channels)
print("notes:", len(notes), notes)
print("velocities:", len(velocities), velocities)

print("done")
print(c, others)
"""