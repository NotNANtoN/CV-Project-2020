# -*- coding: utf-8 -*-

import numpy as np
import wave
import struct

sample_rate = 44100
duration = 5
freq = 440

def save_wave(file_name, audio, rate):
    wave_file=wave.open(file_name,"w")

    # wave params
    nchannels = 1
    sampwidth = 2
    nframes = len(audio)
    comptype = "NONE"
    compname = "not compressed"
    wave_file.setparams((nchannels, sampwidth, rate, nframes, comptype, compname))

    for sample in audio:
        wave_file.writeframes(struct.pack('h', int(sample * 32767 )))

    wave_file.close()

    return
    
# generate sound
t = np.linspace(0., duration, int(sample_rate * duration))
x = np.sin(freq * 2 * np.pi * t)

save_wave("sound.wav", x, sample_rate)