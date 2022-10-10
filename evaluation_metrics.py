# evaluation metrics

"""Integrity related"""

# idea reference: MuseGAN https://github.com/salu133445/musegan/blob/main/src/musegan/metrics.py 
def empty_bar_rate(song_remi_events):
    """
    Return the ratio of empty bars to the total number of bars.
    song_remi_events: the output of generative model, a list of REMI events for one song
    """
    total_num_bars = 0
    empty_bar = 0
    for idx in range(len(song_remi_events)):
      remi_event = song_remi_events[idx]
      if remi_event["name"] == "Bar":
        total_num_bars += 1
        if idx != 0 and song_remi_events[idx - 1]["name"] == "Bar":
          empty_bar += 1
    if total_num_bars == 0:
      raise ValueError("No bars appear in REMI events.")
    return empty_bar / total_num_bars, total_num_bars

# idea reference: MuseGAN
def qualified_note_rate(song_remi_events, threshold=2):
    """Return the ratio of the number of the qualified notes (notes longer than
    `threshold` (in time step)) to the total number of notes in a piano-roll."""
    note_dur = threshold * 120
    qualified_notes = 0
    total_num_notes = 0
    for idx in range(len(song_remi_events)):
      remi_event = song_remi_events[idx]
      if remi_event["name"] == "Note_Duration":
        total_num_notes += 1
        if remi_event["value"] >= note_dur:
          qualified_notes += 1
    if total_num_notes == 0:
      raise ValueError("No notes (Note_Duration) in REMI events.")
    return qualified_notes / total_num_notes

"""Pitch related"""

# code reference: https://github.com/olofmogren/c-rnn-gan/blob/9edafc02e62ea0478d1798bdf3a283c71cd1eefc/midi_statistics.py#L398
# paper: https://arxiv.org/pdf/1611.09904.pdf
def get_pitch_stats(song_remi_events):
  pitches = []
  for idx in range(len(song_remi_events)):
    remi_event = song_remi_events[idx]
    if remi_event["name"] == "Note_Pitch":
      pitches.append(remi_event["value"])
  print(pitches)
  stats = {}
  stats['num_pitches'] = len(pitches)
  stats['pitch_min'] = min(pitches)
  stats['pitch_max'] = max(pitches)
  stats['pitch_span'] = max(pitches)-min(pitches)
  stats['pitches_unique'] = len(set(pitches))
  return stats

# idea reference: MuseGAN
def pitches_per_bar(song_remi_events):
    """Return the number of unique pitches used per bar."""
    total_unique_pitches = set()
    unique_pitches_per_bar = []
    for idx in range(len(song_remi_events)):
      remi_event = song_remi_events[idx]
      if remi_event["name"] == "Bar":
        if idx != 0:
          unique_pitches_per_bar.append(len(pitches_curr_bar))
        pitches_curr_bar = set()
      if remi_event["name"] == "Note_Pitch":
        total_unique_pitches.add(remi_event["value"])
        pitches_curr_bar.add(remi_event["value"])
    return unique_pitches_per_bar


"""Rhythm related"""

# Repetitions of short subsequences of chords were counted, giving a score on how much chord recurrence there is in
# a sample. This metric takes only the tones and their order into account, not their timing.
# idea reference: # C-RNN-GAN https://arxiv.org/pdf/1611.09904.pdf
def chord_repetition(song_remi_events, k=3):
  chords = []
  recurrence = {}
  for idx in range(len(song_remi_events)):
    remi_event = song_remi_events[idx]
    if remi_event["name"] == "Chord":
      chords.append(remi_event["value"])
  total_num_chords = len(chords)
  print(chords)
  for i in range(total_num_chords-k+1):
    pattern = chords[i: i+k]
    for j in range(i+k, total_num_chords-k+1):
      curr_window = chords[j: j+k]
      if pattern == curr_window:
        if tuple(pattern) not in recurrence:
          recurrence[tuple(pattern)] = 1
        else:
          recurrence[tuple(pattern)] += 1
  return recurrence, total_num_chords


# adopted from MuseMorphose 
# code reference: https://github.com/YatingMusic/MuseMorphose/blob/b9a2ac17d996c2b1b395eb5652bc2c30782b7e66/attributes.py#L14
def compute_polyphonicity(song_remi_events):
  n_bars = 0
  for ev in song_remi_events:
    if ev["name"] == "Bar":
      n_bars += 1

  poly_record = np.zeros( (n_bars * 16,) )

  cur_bar, cur_pos = -1, -1
  for ev in song_remi_events:
    if ev['name'] == 'Bar':
      cur_bar += 1
    elif ev['name'] == 'Beat':
      cur_pos = int(ev['value'])
    elif ev['name'] == 'Note_Duration':
      duration = int(ev['value']) // 120
      st = cur_bar * 16 + cur_pos
      poly_record[st:st + duration] += 1
  
  return poly_record


# adopted from MuseMorphose 
# code reference: https://github.com/YatingMusic/MuseMorphose/blob/b9a2ac17d996c2b1b395eb5652bc2c30782b7e66/attributes.py#L14
def get_onsets_timing(song_remi_events):
  n_bars = 0
  for ev in song_remi_events:
    if ev["name"] == "Bar":
      n_bars += 1
      
  onset_record = np.zeros( (n_bars * 16,) )

  cur_bar, cur_pos = -1, -1
  for ev in events:
    if ev['name'] == 'Bar':
      cur_bar += 1
    elif ev['name'] == 'Beat':
      cur_pos = int(ev['value'])
    elif ev['name'] == 'Note_Pitch':
      rec_idx = cur_bar * 16 + cur_pos
      onset_record[ rec_idx ] = 1

  return onset_record
