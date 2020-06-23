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

voc_path = base_dir + "Vocabulary/ORBvoc.txt"
config_path = base_dir + "/Examples/Monocular/TUM1.yaml"

system = System(voc_path, config_path, System.eSensor.MONOCULAR, True)

data_path = "./examples/rgbd_dataset_freiburg1_xyz/"

timestamps = []
image_paths = []
with open(data_path + "rgb.txt") as f:
    for line in f.readlines()[3:]:
        timestamp, img_path = line.split()
        timestamps.append(float(timestamp))
        image_paths.append(img_path)
tframe = timestamps[0]
initialized = False
initialPose = None

for i in range(len(timestamps)):
    #frame = Image.open(data_path + image_paths[i])
    #frame = np.array(frame, dtype=np.float32)
    #frame /= 256.0
    frame = cv2.imread(data_path + image_paths[i], cv2.IMREAD_UNCHANGED)

    t1 = time.time()
    trajectory = system.TrackMonocular(frame, timestamps[i])
    trackingState = system.GetTrackingState()
    if trackingState == 0:
        initialized = False
    elif not initialized and trackingState == 2:
        initialPose = trajectory
        initialized = True
    #print(trajectory)
    t2 = time.time()
    ttrack = t2 - t1
    if i < len(timestamps) - 1:
        T = timestamps[i+1] - tframe
    else:
        T = tframe - timestamps[i-1]

    #if ttrack < T:
    #    time.sleep((T-ttrack))

#mp = system.GetTrackedMapPoints()
#for point in mp:
#    if point is not None:
#        print(point.GetWorldPos())
#        print(point.GetReferenceKeyFrame().GetPose())
Input("Press Enter to continue...")
system.Shutdown()

