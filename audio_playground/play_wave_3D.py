# -*- coding: utf-8 -*-

import openal
import time
import pyautogui

volume_factor = 0.02
pitch_factor = 0.01

source = openal.oalOpen("sound.wav")
source.play()

[mouse_x, mouse_y] = pyautogui.position()

while True:
    
    [curr_x, curr_y] = pyautogui.position()
    
    relative_x, relative_y = curr_x - mouse_x, curr_y - mouse_y
    
    relative_x *= volume_factor
    relative_y *= volume_factor
    
    pitch = 1 - (abs(relative_x) + abs(relative_y)) * pitch_factor
    
    source.set_position([relative_x, relative_y, 0])
    source.set_pitch(pitch)
    
    time.sleep(0.01)
    
    print("Pos(X: {:.2f}, Y: {:.2f}), Pitch({:.2f})".format(relative_x, relative_y, pitch))

    if source.get_state() != openal.AL_PLAYING:
        source.play()

openal.oalQuit()