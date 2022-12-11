import os, pickle, random, copy
import numpy as np

import miditoolkit
import sys

##############################
# use example #
# midi_path = "xxx.mid"
# events = read_generated_txt(midi_path)
# remi2midi(events, note_or_chord="note", output_midi_path="test_note.mid", is_full_event=False)
# remi2midi(events, note_or_chord="chord", output_midi_path="test_chord.mid", is_full_event=False)
##############################



##############################
# constants
##############################
DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
DEFAULT_FRACTION = 16

##############################
# imported from madmom:
# https://github.com/CPJKU/madmom/blob/main/madmom/evaluation/chords.py#L257
##############################

_l = [0, 1, 1, 0, 1, 1, 1]
_chroma_id = (np.arange(len(_l) * 2) + 1) + np.array(_l + _l).cumsum() - 1

NO_CHORD = (-1, -1, np.zeros(12, dtype=int))
UNKNOWN_CHORD = (-1, -1, np.ones(12, dtype=int) * -1)


def pitch(pitch_str):
    """
    Convert a string representation of a pitch class (consisting of root
    note and modifiers) to an integer representation.
    Parameters
    ----------
    pitch_str : str
        String representation of a pitch class.
    Returns
    -------
    pitch : int
        Integer representation of a pitch class.
    """
    return modify(_chroma_id[(ord(pitch_str[0]) - ord('C')) % 7],
                  pitch_str[1:]) % 12

def modify(base_pitch, modifier):
    """
    Modify a pitch class in integer representation by a given modifier string.
    A modifier string can be any sequence of 'b' (one semitone down)
    and '#' (one semitone up).
    Parameters
    ----------
    base_pitch : int
        Pitch class as integer.
    modifier : str
        String of modifiers ('b' or '#').
    Returns
    -------
    modified_pitch : int
        Modified root note.
    """
    for m in modifier:
        if m == 'b':
            base_pitch -= 1
        elif m == '#':
            base_pitch += 1
        else:
            raise ValueError('Unknown modifier: {}'.format(m))
    return base_pitch

def interval(interval_str):
    """
    Convert a string representation of a musical interval into a pitch class
    (e.g. a minor seventh 'b7' into 10, because it is 10 semitones above its
    base note).
    Parameters
    ----------
    interval_str : str
        Musical interval.
    Returns
    -------
    pitch_class : int
        Number of semitones to base note of interval.
    """
    for i, c in enumerate(interval_str):
        if c.isdigit():
            return modify(_chroma_id[int(interval_str[i:]) - 1],
                          interval_str[:i]) % 12
def interval_list(intervals_str, given_pitch_classes=None):
    """
    Convert a list of intervals given as string to a binary pitch class
    representation. For example, 'b3, 5' would become
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0].
    Parameters
    ----------
    intervals_str : str
        List of intervals as comma-separated string (e.g. 'b3, 5').
    given_pitch_classes : None or numpy array
        If None, start with empty pitch class array, if numpy array of length
        12, this array will be modified.
    Returns
    -------
    pitch_classes : numpy array
        Binary pitch class representation of intervals.
    """
    if given_pitch_classes is None:
        given_pitch_classes = np.zeros(12, dtype=int)
    for int_def in intervals_str[1:-1].split(','):
        int_def = int_def.strip()
        if int_def[0] == '*':
            given_pitch_classes[interval(int_def[1:])] = 0
        else:
            given_pitch_classes[interval(int_def)] = 1
    return given_pitch_classes

# mapping of shorthand interval notations to the actual interval representation
_shorthands = {
    'M': interval_list('(1,3,5)'),   # 'maj'
    'm': interval_list('(1,b3,5)'),  # 'min'
    'o': interval_list('(1,b3,b5)'),  # 'dim'
    '+': interval_list('(1,3,#5)'),   # 'aug'
    'M7': interval_list('(1,3,5,7)'),   # 'maj7'
    'm7': interval_list('(1,b3,5,b7)'),  # 'm7'
    '7': interval_list('(1,3,5,b7)'),
    '5': interval_list('(1,5)'),
    '1': interval_list('(1)'),
    'o7': interval_list('(1,b3,b5,bb7)'),  # 'dim7'
    '/o7': interval_list('(1,b3,b5,b7)'),  # 'hdim7'
    'minmaj7': interval_list('(1,b3,5,7)'),
    'maj6': interval_list('(1,3,5,6)'),
    'min6': interval_list('(1,b3,5,6)'),
    '9': interval_list('(1,3,5,b7,9)'),
    'maj9': interval_list('(1,3,5,7,9)'),
    'min9': interval_list('(1,b3,5,b7,9)'),
    'sus2': interval_list('(1,2,5)'),
    'sus4': interval_list('(1,4,5)'),
    '11': interval_list('(1,3,5,b7,9,11)'),
    'min11': interval_list('(1,b3,5,b7,9,11)'),
    '13': interval_list('(1,3,5,b7,13)'),
    'maj13': interval_list('(1,3,5,7,13)'),
    'min13': interval_list('(1,b3,5,b7,13)')
}


def chord_intervals(quality_str):
    """
    Convert a chord quality string to a pitch class representation. For
    example, 'maj' becomes [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0].
    Parameters
    ----------
    quality_str : str
        String defining the chord quality.
    Returns
    -------
    pitch_classes : numpy array
        Binary pitch class representation of chord quality.
    """
    list_idx = quality_str.find('(')
    if list_idx == -1:
        return _shorthands[quality_str].copy()
    if list_idx != 0:
        ivs = _shorthands[quality_str[:list_idx]].copy()
    else:
        ivs = np.zeros(12, dtype=int)

    return interval_list(quality_str[list_idx:], ivs)


def chord(label):
    """
    Transform a chord label into the internal numeric representation of
    (root, bass, intervals array) as defined by `CHORD_DTYPE`.
    Parameters
    ----------
    label : str
        Chord label.
    Returns
    -------
    chord : tuple
        Numeric representation of the chord: (root, bass, intervals array).
    """
    if label == 'N':
        return NO_CHORD
    if label == 'X':
        return UNKNOWN_CHORD

    c_idx = label.find(':')
    # s_idx = label.find('/')
    s_idx = -1

    if c_idx == -1:
        quality_str = 'maj'
        if s_idx == -1:
            root_str = label
            bass_str = ''
        else:
            root_str = label[:s_idx]
            bass_str = label[s_idx + 1:]
    else:
        root_str = label[:c_idx]
        if s_idx == -1:
            quality_str = label[c_idx + 1:]
            bass_str = ''
        else:
            quality_str = label[c_idx + 1:s_idx]
            bass_str = label[s_idx + 1:]

    root = pitch(root_str)
    bass = interval(bass_str) if bass_str else 0
    ivs = chord_intervals(quality_str)
    ivs[bass] = 1

    return root, bass, ivs

##############################
# containers for conversion
##############################
class ConversionEvent(object):
    def __init__(self, event, is_full_event=False):
        if not is_full_event:
            if 'Note' in event:
                self.name, self.value = '_'.join(event.split('_')[:-1]), event.split('_')[-1]
            elif 'Chord' in event:
                self.name, self.value = event.split('_')[0], '_'.join(event.split('_')[1:])
            else:
                self.name, self.value = event.split('_')
        else:
            self.name, self.value = event['name'], event['value']
    def __repr__(self):
        return 'Event(name: {} | value: {})'.format(self.name, self.value)

class NoteEvent(object):
    def __init__(self, pitch, bar, position, duration, velocity):
        self.pitch = pitch
        self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)
        self.duration = duration
        self.velocity = velocity
  
class TempoEvent(object):
    def __init__(self, tempo, bar, position):
        self.tempo = tempo
        self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)

class ChordEvent(object):
    def __init__(self, chord_val, bar, position):
        self.chord_val = chord_val
        self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)


##############################
# defined to fit our own tasks
##############################


def chord2notes(chord_name, prev_pitch):
    # adjust root note to be closest to prev_pitch
    # chord ex: C_sus4, C#_7
    root, chord_type = chord_name.split("_")
    converted = ":".join([root, chord_type])
    root, bass, ivs = chord(converted)
    # convert to several pitches and return
    res = []
    for i in range(len(ivs)):
        if ivs[i] == 1:
            res.append(root + i)
    if prev_pitch > root:
        for i in range(10):
            if i * 12 + root <= prev_pitch and (i+1) * 12 + root > prev_pitch:
                for k in range(len(res)):
                    res[k] += i * 12
                break
    # print(res)
    if res[-1] > 127:
        for i in range(len(res)):
            res[i] -= 12
    return res


def adjust_pitch(note_pitch, base_pitch):
    while note_pitch < base_pitch - 2 * 12:
        note_pitch += 12
    while note_pitch > base_pitch - 2 * 12:
        note_pitch -= 12
    return note_pitch


##############################
# conversion functions
##############################
def read_generated_txt(generated_path):
    f = open(generated_path, 'r')
    return f.read().splitlines()

def remi2midi(events, output_midi_path=None, note_or_chord="note", is_full_event=False, return_first_tempo=False, enforce_tempo=False, enforce_tempo_val=None):
    events = [ConversionEvent(ev, is_full_event=is_full_event) for ev in events]
    # print (events[:20])

    assert events[0].name == 'Bar'
    temp_notes = []
    temp_tempos = []
    temp_chords = []

    used_chords_n_pitches = []
    cur_bar = 0
    cur_position = 0
    # base_pitch = 66
    prev_pitch = 25
    prev_velocity = 0
    chord_in_curr_beat = False   # max 1 chord per beat

    for i in range(len(events)):
        if events[i].name == 'Bar':
            chord_in_curr_beat = False
            if i > 0:
                cur_bar += 1
        elif events[i].name == 'Beat':
            chord_in_curr_beat = False
            cur_position = int(events[i].value)
            assert cur_position >= 0 and cur_position < DEFAULT_FRACTION
        elif events[i].name == 'Tempo':
            temp_tempos.append(TempoEvent(
                int(events[i].value), cur_bar, cur_position
            ))
        elif 'Note_Pitch' in events[i].name and \
               (i+1) < len(events) and 'Note_Velocity' in events[i+1].name and \
               (i+2) < len(events) and 'Note_Duration' in events[i+2].name:
            # check if the 3 events are of the same instrument
            # print("current position:", cur_position)  # 0 - 15
            # print(int(events[i].value))
            curr_pitch = int(events[i].value) # adjust_pitch(int(events[i].value), base_pitch)
            used_chords_n_pitches.append(curr_pitch)
            prev_pitch = curr_pitch
            prev_velocity = int(events[i+1].value)
            # print("duration:", int(events[i+2].value))
            if note_or_chord == "note":
                temp_notes.append(
                    NoteEvent(
                        pitch=curr_pitch, 
                        bar=cur_bar, position=cur_position, 
                        duration=int(events[i+2].value), velocity=int(events[i+1].value)
                    )
                )
        elif 'Chord' in events[i].name and note_or_chord == "chord":
            if chord_in_curr_beat:
                continue
            if events[i].value[-1] == "N":
                continue
            # print(prev_pitch)
            used_chords_n_pitches.append(events[i].value)
            chord_notes = chord2notes(events[i].value, prev_pitch)  # prev_pitch
            for nt in chord_notes:
                temp_notes.append(
                    NoteEvent(
                        pitch=nt, # adjust_pitch(nt, base_pitch), 
                        bar=cur_bar, position=cur_position, 
                        duration=240, velocity=prev_velocity
                    )
                )
            chord_in_curr_beat = True
        elif events[i].name in ['EOS', 'PAD']:
            continue


    # print (len(temp_tempos), len(temp_notes))
    midi_obj = miditoolkit.midi.parser.MidiFile()
    midi_obj.instruments = [
        miditoolkit.Instrument(program=0, is_drum=False, name='Piano')
    ]

    for n in temp_notes:
        midi_obj.instruments[0].notes.append(
            miditoolkit.Note(int(n.velocity), n.pitch, int(n.start_tick), int(n.start_tick + n.duration))
        )

    if enforce_tempo is False:
        for t in temp_tempos:
            midi_obj.tempo_changes.append(
                miditoolkit.TempoChange(t.tempo, int(t.start_tick))
            )
    else:
        if enforce_tempo_val is None:
            enforce_tempo_val = temp_tempos[1]
        for t in enforce_tempo_val:
            midi_obj.tempo_changes.append(
                miditoolkit.TempoChange(t.tempo, int(t.start_tick))
            )

    
    for c in temp_chords:
        # print(c.chord_val, c.start_tick)
        midi_obj.markers.append(
            miditoolkit.Marker('Chord-{}'.format(c.chord_val), int(c.start_tick))
        )
    for b in range(cur_bar):
        # print(DEFAULT_BAR_RESOL * b)
        midi_obj.markers.append(
            miditoolkit.Marker('Bar-{}'.format(b+1), int(DEFAULT_BAR_RESOL * b))
        )

    # print("used", used_chords_n_pitches)
    
    if output_midi_path is not None:
        midi_obj.dump(output_midi_path)

    if not return_first_tempo:
        return midi_obj
    else:
        return midi_obj, temp_tempos


if __name__ == "__main__":
    midi_path = sys.argv[1]
    events = read_generated_txt(midi_path)
    remi2midi(events, note_or_chord="note", output_midi_path="test_note.mid", is_full_event=False)
    remi2midi(events, note_or_chord="chord", output_midi_path="test_chord.mid", is_full_event=False)



