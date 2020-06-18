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
from object_detection.detect_bananas import YOLO
from audio_playground.Audio import Audio
from DenseDepth.monodepth import MonoDepth

# Read intrinsic camera parameters, if none detected prompt calibration.
try:
    camera_matrix = np.load("calibration/camera_matrix.npy")
    dist_coefs = np.load("calibration/dist_coefs.npy")
except FileNotFoundError:
    print("Calibration parameter loading failed. Please go to the folder calibrate/ to calibrate your camera.")
    quit()

parser = argparse.ArgumentParser()
parser.add_argument("--mono", default=0, type=int, help="Whether to use monocular depth estimation")
parser.add_argument("-s", help="Data source. Either cam or path to data",
                    default="object_detection/input/video/ycb_seq1_fast.mp4", type=str)
parser.add_argument("--tiny", type=int, default=1, help="Whether to use the tiny yolo model instead of the large one")
args = parser.parse_args()

# Instantiate all algorithms
use_mono_depth = args.mono
if args.s == "cam":
    cam = Webcam()
else:
    cam = VideoInput(args.s)
yolo = YOLO(path_extension="object_detection", use_tiny=args.tiny)
if use_mono_depth:
    depth_model = MonoDepth("DenseDepth/", parser=parser)
audio = Audio("audio_playground/sound.wav")

# Create Camera object
# Get camera feed from camera object
cam.start()

def get_position_bbox(img, out_boxes):
    top, left, bottom, right = out_boxes[0]
    center_x = (right - left) / 2 + left
    center_y = (bottom - top) / 2 + top

    im_width, im_height, _ = img.shape

    pos_x = (center_x - im_width / 2) / im_width
    pos_y = (center_y - im_height / 2) / im_height
    left, right, top, bottom = int(left), int(right), int(top), int(bottom)
    print("coords: ", left, right, top, bottom)
    print(im_width, im_height)

    return [pos_x, pos_y, 1.0]

def get_depth(depth_map):
    depth_box = depth_map[top:bottom, left:right]
    print("Depth box shape: ", depth_box.shape)
    cv2.imshow("Depth box of object", depth_box)
    pos_z = depth_box.mean()
    cv2.imshow("Depth box in orig img", img[top:bottom, left:right, :])

    return pos_z

def process_frame(frame):
    frame_np = np.array(frame)
    # First, calibrate the frame:
    frame_np = undistort_image(frame_np, camera_matrix, dist_coefs)
    frame_PIL = Image.fromarray(frame_np)

    # Feed camera feed into object detection algorithm to get bounding boxes
    # Show bounding boxes in feed
    yolo_image, out_boxes, out_scores = yolo.detect_image(frame_PIL)
    yolo_image = np.array(yolo_image)

    #yolo_image.save(dir_output + '/yolo_' + file_img)
                    
    #for i in range(len(out_boxes)): 
    #    top, left, bottom, right = out_boxes[i]
    #    file_detections.write('Banana {} in {}: Top: {}, Left: {}, Bottom: {}, Right: {}, Confidence: {}\n'.format(i+1, file_img, top, left, bottom, right, out_scores[i]))      
    if out_boxes:
        cv2.imshow("Auralizer", yolo_image)

        # Combine bounding box and depth to get coordinate of object.
        object_position = get_position_bbox(yolo_image, out_boxes)

        # METHOD 1 - Depth estimation:
        # Feed camera feed into monocular depth estimation algorithm and get depth map
        # Show depth map
        if use_mono_depth:
            start_time = time.time()
            depth_map = depth_model.forward(frame_np)
            print("Time Depth Est: ", round(time.time() - start_time, 1))
            depth_map = depth_map.squeeze()
            # Upsample the depth map:
            height, width = frame_np.shape[:2]
            depth_map = np.array(Image.fromarray(depth_map).resize((width, height)))
            cv2.imshow("Depth image afterwards", depth_map)
            object_position[2] = get_depth(depth_map)

        print("Object pos: ", object_position)

        # METHOD 2 - 
        # Feed camera feed into SLAM and get list of features with coordinates
        # Mark features that can be seen from current frame and that are within bounding box as relating to the object
        # Get coordinates of the feature group that is closest and/or has the highest density of detected features for the sought object

        # METHOD 3 -
        # Calculate center of detected bbox relative to camera center
        # Those coordinates will represent x and y of the current 3D position (with fixed z value)

        # FINAL:
        # Give coordinate of object to auralizer to create sound.

        audio.play()
        audio.set_position(object_position)
        print()
    else:
        cv2.imshow("Auralizer", frame_np)

while True:
    frame = cam.get_current_frame()
    if frame is not None:
        process_frame(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
