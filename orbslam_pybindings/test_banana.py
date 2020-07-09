import sys
import os
import time
import numpy as np
import cv2
from PIL import Image
base_dir = "../ORB_SLAM2/"
sys.path.insert(0, "./lib/")
from orbslam2 import System
from orbslam2 import MapPoint

def ComputeRealWorldCoordinates(relPos, scale):
    return relPos * scale

voc_path = base_dir + "Vocabulary/ORBvoc.txt"
#config_path = base_dir + "/Examples/Monocular/TUM1.yaml"
config_path = "../calibration/orbslam_config.yaml"

system = System(voc_path, config_path, System.eSensor.MONOCULAR, True)

data_path = "../object_detection/input/"

image_paths = []
for f in os.listdir(data_path):
    if not f == "video":
        for _ in range(10):
            image_paths.append(f)

initialized = False
initialPose = None

cap = cv2.VideoCapture(data_path + 'video/ycb_seq1.mp4')
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break
    trajectory = system.TrackMonocular(frame, time.time())
    trackingState = system.GetTrackingState()
    if trackingState == 3 or trackingState == 1:
        initialized = False
    elif not initialized and trackingState == 2:
        mps = system.GetTrackedMapPoints()
        for mp in mps:
            if not mp is None:
                kf = mp.GetReferenceKeyFrame()
                initial_pose = kf.GetPoseInverse()
                print(initialPose)
                scale = kf.ComputeSceneMedianDepth()
                break
        initialized = True
    elif initialized and trackingState == 2:
        coordinates = system.GetTrackedMapPointsPositions()
        print(coordinates)
    print(trajectory)
cap.release()

mp = system.GetTrackedMapPoints()
for point in mp:
    if point is not None:
        worldpos = point.GetWorldPos()
        realPos = ComputeRealWorldCoordinates(worldpos, scale)
        print(worldpos)
        print(realPos)
        print("")
input("Press Enter to continue...")
system.Shutdown()
