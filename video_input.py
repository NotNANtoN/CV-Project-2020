# -*- coding: utf-8 -*-

import cv2
from PIL import Image

class VideoInput():
    def __init__(self, video_file):
        self.video_file = video_file

    def start(self):
        self.video = cv2.VideoCapture(self.video_file)
        if not self.video.isOpened():
            raise IOError("Couldn't open video")

    def get_current_frame(self):
        _, frame = self.video.read()
        if frame:
            frame = Image.fromarray(frame)
        else:
            return None
        return frame