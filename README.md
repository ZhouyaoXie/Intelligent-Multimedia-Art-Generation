# Intelligent-Multimedia-Art-Generation
Capstone project for CMU MCDS program

# Current Dataset info
Number of unique recordings: 9409

Number of music chunks (splitted every 16 bar): 34241 - 2255 (files that have no events between at least one of the two consecutive bars) = 31986

Average number of bars per chunks (among 34241 files): 13.84

dataset split | # positive pairs | # negative pairs
--- | --- | --- 
train | 544785 | 544785
val | 54914 | 54914
test | 69776 | 69776

# Bar Position

type | Number of events between two consecutive bars
--- | --- 
min | 0
max| 2130
avg | 58.97

Number of events between two consecutive bars (range) | frequency (times appeared)
--- | --- 
0 - 99 | 358354
100 - 199| 57471
200 - 299 | 14875
300 - 399 | 5028
400 - 499 | 2002
500 - 599 | 929
600 - 699 | 552
700 - 799 | 122
800 - 899 | 127
900 - 999 | 23
1000 - 1099 | 22
1100 - 1199 | 11
1200 - 1299 | 9
1300 - 1399 | 1
1400 - 1499 | 1
1500 - 1599 | 0
1600 - 1699 | 0
1700 - 1799 | 0
1800 - 1899 | 0
1900 - 1999 | 0
2000 - 2099 | 12
2100 - 2199 | 4

# Attribute Class

rhythmic intensity bounds: [0.0, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.9375, 1.0]
polyphonicity bounds: [0.0, 0.5, 0.75, 0.9375, 1.25, 2.0, 3.0, 14.0]

[rhythm classes] Counter({0: 2917246, 4: 2615224, 1: 2537060, 2: 2413000, 7: 2188198, 3: 2018130, 5: 2005684, 6: 1769110})
[polyph classes] Counter({0: 5395600, 2: 2086768, 6: 2081836, 1: 1977004, 3: 1824338, 4: 1772322, 5: 1759396, 7: 1566388})

# REMI Vocabulary
 ```
 {
   'Bar': 1,
   'Beat': 16,
   'Chord': 133,
   'EOS': 1,
   'Note_Duration': 17,
   'Note_Pitch': 126,
   'Note_Velocity': 44,
   'Tempo': 65
 }
 ```
Chord specific vocabluary:
 ```
 {
  'A#': 11,
  'A': 11,
  'B': 11,
  'C#': 11,
  'C': 11,
  'D#': 11,
  'D': 11,
  'E': 11,
  'F#': 11,
  'F': 11,
  'G#': 11,
  'G': 11,
  'N': 1
 }
 ```
Note Duration statistics:
 ```
{
  240: 779506,
  120: 1807302,
  960: 41849,
  840: 21399,
  360: 250845,
  1920: 31949,
  480: 179433,
  0: 4955063,
  600: 66738,
  720: 72965,
  1080: 13200,
  1320: 8336,
  1560: 15080,
  1440: 17063,
  1200: 14781,
  1800: 4310,
  1680: 4311
}
```
 
# Sample REMI events
```
Event(name=Bar, time=None, value=None, text=1)
Event(name=Position, time=0, value=1/16, text=0)
Event(name=Chord, time=0, value=F:maj, text=F:maj)
Event(name=Position, time=0, value=1/16, text=0)
Event(name=Tempo Class, time=0, value=mid, text=None)
Event(name=Tempo Value, time=0, value=30, text=None)
Event(name=Position, time=0, value=1/16, text=0)
Event(name=Note Velocity, time=0, value=25, text=100/100)
Event(name=Note On, time=0, value=69, text=69)
Event(name=Note Duration, time=0, value=0, text=64/60)
Event(name=Position, time=120, value=2/16, text=120)
Event(name=Note Velocity, time=120, value=25, text=100/100)
Event(name=Note On, time=120, value=76, text=76)
Event(name=Note Duration, time=120, value=0, text=70/60)
Event(name=Position, time=240, value=3/16, text=240)
Event(name=Note Velocity, time=240, value=25, text=100/100)
Event(name=Note On, time=240, value=74, text=74)
Event(name=Note Duration, time=240, value=0, text=70/60)
Event(name=Position, time=240, value=3/16, text=240)
Event(name=Note Velocity, time=240, value=25, text=100/100)
Event(name=Note On, time=240, value=72, text=72)
Event(name=Note Duration, time=240, value=0, text=53/60)
Event(name=Position, time=360, value=4/16, text=360)
Event(name=Note Velocity, time=360, value=25, text=100/100)
Event(name=Note On, time=360, value=69, text=69)
Event(name=Note Duration, time=360, value=0, text=66/60)
Event(name=Position, time=480, value=5/16, text=480)
Event(name=Note Velocity, time=480, value=25, text=100/100)
Event(name=Note On, time=480, value=76, text=76)
Event(name=Note Duration, time=480, value=0, text=72/60)
```

# References
- midi2remi.ipynb: from https://github.com/YatingMusic/remi
- utils.py: from https://github.com/YatingMusic/remi
- chord_recognition.py: from https://github.com/YatingMusic/remi
