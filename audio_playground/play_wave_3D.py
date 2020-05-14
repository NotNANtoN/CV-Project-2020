# -*- coding: utf-8 -*-

import openal
import time
import pyautogui
from msvcrt import getch, kbhit

volume_factor = 0.02
pitch_factor = 0.01

z_pos = 0.0
z_speed = 0.5

source = openal.oalOpen("sound.wav")
source.play()

[mouse_x, mouse_y] = pyautogui.position()

while True:
    if kbhit():
        key = ord(getch())

        if key == 80: #down-arrow
                z_pos += z_speed
        elif key == 72: #up-arrow
                z_pos -= z_speed
    
    [curr_x, curr_y] = pyautogui.position()
    
    relative_x, relative_y = curr_x - mouse_x, curr_y - mouse_y

    x_pos, y_pos = relative_x * mouse_scale, relative_y * mouse_scale
    
<<<<<<< HEAD
    relative_x *= volume_factor
    relative_y *= volume_factor
    
    pitch = 1 - (abs(relative_x) + abs(relative_y)) * pitch_factor
    
    source.set_position([relative_x, relative_y, 0])
    source.set_pitch(pitch)
    
    time.sleep(0.01)
    
    print("Pos(X: {:.2f}, Y: {:.2f}), Pitch({:.2f})".format(relative_x, relative_y, pitch))
=======
    source.set_position([x_pos, y_pos, z_pos])
    source.set_pitch(1 + (abs(relative_x) + abs(relative_y)) * pitch_factor)
    
    time.sleep(0.01)
    
    print("X: {}, Y: {}, Z: {}".format(x_pos, y_pos, z_pos))
>>>>>>> 54e0f949f2cae650eb19a598fb3d9cc79ef30adc

    if source.get_state() != openal.AL_PLAYING:
        source.play()

openal.oalQuit()
