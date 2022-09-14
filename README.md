# Intelligent-Multimedia-Art-Generation
Capstone project for CMU MCDS program

# Current Dataset info

dataset split | # positive pairs | # negative pairs
--- | --- | --- 
train | 574999 | 574999
val | 80008 | 80008
test | 63241 | 63241

# Attribute Class

rhythm classes:  `Counter({0: 153656, 7: 115504, 6: 50451, 5: 41513, 2: 32963, 1: 31840, 3: 26032, 4: 21825})`

polyph classes: `Counter({0: 424411, 1: 16466, 2: 11606, 3: 10113, 4: 5512, 5: 3749, 6: 1552, 7: 375})`

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
