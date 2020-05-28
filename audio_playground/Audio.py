# -*- coding: utf-8 -*-

import openal
import numpy as np

class Audio():
    def __init__(self, wavefile, volume_factor=0.2, pitch_factor=0.01, base_pitch=1):
        self.source = openal.oalOpen(wavefile)
        self.volume_factor = volume_factor
        self.pitch_factor = pitch_factor
        self.base_pitch = base_pitch

    def __del__(self):
        openal.oalQuit()

    def play(self):
        if self.source.get_state() != openal.AL_PLAYING:
            self.source.play()

    def set_position(self, position):
        self.source.set_position(position)

        # Calculate angle between object and camera
        angle = self.angle(position, [0, 0, 1])

        pitch = self.base_pitch + angle * self.pitch_factor
        self.source.set_pitch(pitch)

    def mute(self):
        self.source.set_gain(0)

    def unmute(self):
        self.source.set_gain(1)

    def angle(self, v1, v2):
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
