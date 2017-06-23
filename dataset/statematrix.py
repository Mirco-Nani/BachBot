from __future__ import print_function
import midi, numpy
from os import listdir
from os.path import isfile, join

lowerBound = 24
upperBound = 102

def midiToNoteStateMatrix(midifile):
    valid = True

    pattern = midi.read_midifile(midifile)

    timeleft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]

    statematrix = []
    span = upperBound-lowerBound
    time = 0

    state = [[0,0] for x in range(span)]
    statematrix.append(state)
    while True:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(span)]
            statematrix.append(state)

        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                        print("Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time))
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lowerBound] = [0, 0]
                        else:
                            state[evt.pitch-lowerBound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        print("Found time signature event {}. Bailing!".format(evt))
                        return statematrix, False

                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            #print("all(t is None for t in timeleft)")
            valid = False
            break

        time += 1

    return statematrix, valid


source = "files/bach/flattened"

midis = [(f,join(source, f)) for f in listdir(source) if isfile(join(source, f))]

found = False

for filename,path in midis:
    print(filename,":")
    statematrix, valid = midiToNoteStateMatrix(path)
    if(valid):
        found = True
        print(len(statematrix))
        print(len(statematrix[0]))
        print(len(statematrix[0][0]))
        #print(len(statematrix[0][0][0]))
        break
print(found)
#statematrix, valid = midiToNoteStateMatrix("files/mir1.mid")


"""
statematrix, valid = midiToNoteStateMatrix("files/bach/flattened/01allema.mid")

print(len(statematrix))
print(len(statematrix[0]))
print(len(statematrix[0][0]))


print(statematrix)
"""

print("done")