# -*- coding: utf-8 -*-

import openal
import time

source = openal.oalOpen("sound.wav")

source.set_pitch(1)
source.set_gain(1)
source.play()

position_x = -5

while source.get_state() == openal.AL_PLAYING:
    position_x += .1
    source.set_position([position_x, 0, 0])
    time.sleep(0.05)
    print("Position X: {:.2f}".format(position_x))

openal.oalQuit()