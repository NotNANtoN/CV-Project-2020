import numpy as np
import cv2
import argparse
import sys
import time
sys.path.insert(0, "./DenseDepth/")

from PIL import Image

from calibration.webcam import Webcam
from calibration.calibrate import undistort_image
from video_input import VideoInput
from object_detection.detect_YCB import YOLO
from audio_playground.Audio import Audio
from DenseDepth.monodepth import MonoDepth
from bts_depth.bts.pytorch.bts_modular import BTS
from orbslam_pybindings.slam import ORBSLAM2

# Read intrinsic camera parameters, if none detected prompt calibration.
try:
    camera_matrix = np.load("calibration/camera_matrix.npy")
    dist_coefs = np.load("calibration/dist_coefs.npy")
except FileNotFoundError:
    print("Calibration parameter loading failed. Please go to the folder calibrate/ to calibrate your camera.")
    quit()

parser = argparse.ArgumentParser()
parser.add_argument("-s", default="object_detection/input/video/ycb_seq1.mp4", type=str, help="Data source. Either cam or path to data.")
parser.add_argument("--mono", choices=["none", "mono", "bts"], default="none", help="Whether to use monocular depth estimation.")
parser.add_argument("--mp", default=0, type=int, help="Whether to use multiprocessing for the camera input. Not working for windows OS.")
parser.add_argument("--yolomodel", default=0, type=int, help="Whether to use YCB-model (0) or COCO (1).")
parser.add_argument("--object", default=0, type=int, help="Which object to search for.")
parser.add_argument("--slam", default=0, type=int, help="Whether to use slam")
args = parser.parse_args()

# Instantiate all algorithms
use_mono_depth = args.mono != "none"
if args.s == "cam":
    cam = Webcam(sequential=not args.mp)
else:
    cam = VideoInput(args.s)

yolo = YOLO(path_extension="object_detection", model=args.yolomodel)
audio = Audio("audio_playground/sound.wav")

if use_mono_depth:
    if args.mono == "mono":
        depth_model = MonoDepth("DenseDepth/", parser=parser)
    else:
        depth_model = BTS(parser)

use_slam = args.slam
if use_slam:
    slam = ORBSLAM2(useViewer=False)

# define initial object class to search for: 0 = meat can, 1 = banana, 2 = large marker
search_object_class = args.object
num_classes = yolo.get_num_classes()

# Create Camera object
# Get camera feed from camera object
cam.start()

def get_position_bbox(img, out_box):
    top, left, bottom, right = out_box
    center_x = (right - left) / 2 + left
    center_y = (bottom - top) / 2 + top

    im_width, im_height, _ = img.shape

    pos_x = (center_x - im_width / 2) / im_width
    pos_y = (center_y - im_height / 2) / im_height
    left, right, top, bottom = int(left), int(right), int(top), int(bottom)
    #print("coords: ", left, right, top, bottom)
    #print(im_width, im_height)

    return [pos_x, pos_y, 1.0]

def get_depth(depth_map, out_box):
    top, left, bottom, right = map(lambda x: int(x), out_box)
    depth_box = depth_map[top:bottom, left:right]
    #print("Depth box shape: ", depth_box.shape)
    cv2.imshow("Depth box of object", depth_box)
    pos_z = depth_box.mean()
    #cv2.imshow("Depth box in orig img", img[top:bottom, left:right, :])

    return pos_z

def process_frame(frame):
    frame_np = np.array(Image.fromarray(frame))
    # First, calibrate the frame:
    frame_np = undistort_image(frame_np, camera_matrix, dist_coefs)
    height, width = frame_np.shape[:2]
    frame_PIL = Image.fromarray(frame_np)

    # Feed camera feed into object detection algorithm to get bounding boxes
    # Show bounding boxes in feed
    yolo_image, out_box = yolo.detect_image(frame_PIL, search_object_class)
    yolo_image = np.array(yolo_image)

    if use_slam:
        slam.track_monocular(frame, time.time())
        print("slam pose estimation: {}".format(slam.pose))
        if slam.initialized:
            yolo_image = slam.draw_keypoints(yolo_image)
            print("Scale: {}".format(slam.scale))

    cv2.imshow("Auralizer", yolo_image)

    if out_box is not None:

        # Combine bounding box and depth to get coordinate of object.
        object_position = get_position_bbox(yolo_image, out_box)

        # METHOD 1 - Depth estimation:
        # Feed camera feed into monocular depth estimation algorithm and get depth map
        # Show depth map
        if use_mono_depth:
            start_time = time.time()
            depth_map = depth_model.forward(frame_np)
            #print("Time Depth Est: ", round(time.time() - start_time, 1))
            depth_map = depth_map.squeeze()
            # Upsample the depth map:
            if args.mono == "bts":
                depth_map = depth_map.numpy()
            else:
                depth_map = np.array(Image.fromarray(depth_map).resize((width, height)))
            #print("mean: ", depth_map.mean())
            #print("std: ", depth_map.std())
            #print("section: ", depth_map[:10, :10])
            #print("depth map shape: ", depth_map.shape)
            #cv2.imshow("Full depth image", depth_map)
            # normalize depth map:
            #depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            #cv2.imshow("Normed depth image", depth_map)
            # get distance by averaging over depth in depth_box
            distance = get_depth(depth_map, out_box)
            # scale distance for better volume behavior
            print("Distance before scaling: ", distance)
            if args.mono == "bts":
                distance = max(distance - 13, 0) * 0.3
            else:
                distance = max(distance - 0.05, 0) * 300
            object_position[2] = distance
            print("Depth perception time: ", round(time.time() - start_time, 2))
        else:
            top, left, bottom, right = out_box
            # calc box area:
            box_area = (top - bottom) * (left - right)
            # get proportion to img:
            proportion = box_area / (height * width)
            # get distance
            distance = 1 - proportion
            # scale distance for better volume behavior
            # (distance is in range 0.5-0.8, after scaling in 0-6)
            print("Distance before scaling: ", distance)
            distance = max((distance - 0.5), 0) * 20
            # assign to distance:
            object_position[2] = distance
        print("Predicted distance: ", object_position[2])


        #print("Object pos: ", object_position)

        # METHOD 2 -
        # Feed camera feed into SLAM and get list of features with coordinates
        # Mark features that can be seen from current frame and that are within bounding box as relating to the object
        # Get coordinates of the feature group that is closest and/or has the highest density of detected features for the sought object

        if use_slam and slam.initialized:
            object_position = slam.compute_object_position(out_box)
            print(object_position)
            #if not use_mono_depth:
            #    object_position[2] = slam_object_position[2]

        audio.play()
        audio.set_position(object_position)

        # METHOD 3 -
        # Calculate center of detected bbox relative to camera center
        # Those coordinates will represent x and y of the current 3D position (with fixed z value)

        # FINAL:
        # Give coordinate of object to auralizer to create sound.

        print()
    elif use_slam:
        if slam.last_tracked_object is not None:
            object_position = slam.compute_last_tracked_object_position()
            print("Last tracked object: {}".format(object_position))
            audio.play()
            audio.set_position(object_position)
            print()

def clean_up():
    audio.__del__()

while True:
    frame = cam.get_current_frame()
    if frame is not None:
        process_frame(frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        clean_up()
        break

    # Count up or down
    if key == ord('1') or key == ord('2'):
        if key == ord('1'):
            search_object_class -= 1
        if key == ord('2'):
            search_object_class += 1

        search_object_class = search_object_class % (num_classes - 1)
        audio.stop()
