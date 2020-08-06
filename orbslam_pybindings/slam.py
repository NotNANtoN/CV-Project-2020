"""
Class definition of ORB_SLAM2 system
"""

import sys
import os
import time
import numpy as np
import cv2
import PIL
sys.path.insert(0, "./ORB_SLAM2/lib/")
base_dir = "./ORB_SLAM2/"
from orbslam2 import System
from orbslam2 import MapPoint
from orbslam2 import KeyFrame

class ORBSLAM2:
    """docstring for ORBSLAM2."""

    def __init__(self, useViewer=True):
        #super(ORBSLAM2 , self).__init__()
        self.useViewer = useViewer

        voc_path = base_dir + "Vocabulary/ORBvoc.txt"
        config_path = "./calibration/orbslam_config.yaml"

        self.system = System(voc_path, config_path, System.eSensor.MONOCULAR, useViewer)
        self.initialized = False
        self.scale = None
        self.pose = None
        self.last_tracked_object = None
        self.object_mappoints = []
        self.object_keypoints = []
        self.tracked_key_points = []
        self.tracked_map_points = []

    def track_monocular(self, frame, timestamp):
        self.pose = self.system.TrackMonocular(frame, timestamp)
        trackingState = self.system.GetTrackingState()
        if trackingState == 2 and not self.initialized:
            self.initialized = True
        elif self.initialized and trackingState != 2:
            self.initialized = False
            self.last_tracked_object = None
        if self.initialized:
            self.scale = self.system.GetScale()
            self.tracked_map_points = np.squeeze(np.array(self.system.GetTrackedMapPointsPositions()))
            self.tracked_key_points = np.squeeze(np.array(self.system.GetTrackedKeyPointsUnPositions()))

    def point_inside_box(self, point, out_box):
        top, left, bottom, right = out_box
        x,y = point
        return x >= left and x <= right and y >= top and y <= bottom

    def compute_object_position(self, out_box):
        self.object_mappoints = np.array([mp for mp,kp in zip(self.tracked_map_points, self.tracked_key_points) if self.point_inside_box(kp, out_box)])
        self.object_keypoints = np.array([kp for kp in self.tracked_key_points if self.point_inside_box(kp, out_box)])
        self.last_tracked_object = np.append(np.average(self.object_mappoints, axis=0),[1])[None].T
        object_position = np.matmul(np.linalg.inv(self.pose), self.last_tracked_object)
        return np.squeeze(object_position[:-1])

    def compute_last_tracked_object_position(self):
        object_position = np.matmul(np.linalg.inv(self.pose), self.last_tracked_object)
        return np.squeeze(object_position[:-1])

    def draw_keypoints(self, frame):
        frame = PIL.Image.fromarray(frame)
        draw = PIL.ImageDraw.Draw(frame)
        for kp in self.tracked_key_points:
            draw.ellipse((kp[0] - 2, kp[1] - 2, kp[0] + 2, kp[1] + 2), fill = 'green')
        return np.array(frame)
