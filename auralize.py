import numpy as np
import cv2
import argparse
import sys
sys.path.insert(0, "./DenseDepth/")

from PIL import Image

from calibration.webcam import Webcam
from video_input import VideoInput
from object_detection.detect_bananas import YOLO
from audio_playground.Audio import Audio
from DenseDepth.monodepth import MonoDepth

# Read intrinsic camera parameters, if none detected prompt calibration.
#try:
camera_matrix = np.load("calibration/camera_matrix.npy")
dist_coefs = np.load("calibration/dist_coefs.npy")

parser = argparse.ArgumentParser()
parser.add_argument("-s", help="Data source. Either cam or path to data", default="object_detection/input/video/ycb_seq1_fast.mp4", type=str)
args = parser.parse_args()

# Instantiate all algorithms

if args.s == "cam":
    cam = Webcam()
else:
    cam = VideoInput(args.s)
yolo = YOLO("object_detection")  # assumes YOLO model weights are downloaded! (see keras yolo readme)
depth_model = MonoDepth("DenseDepth/", parser=parser)
audio = Audio("audio_playground/sound.wav")

# Create Camera object
# Get camera feed from camera object
cam.start()

def get_position_bbox(img, out_boxes, depth_map):
    top, left, bottom, right = out_boxes[0]
    center_x = (right - left) / 2 + left
    center_y = (bottom - top) / 2 + top

    im_width, im_height, _ = img.shape

    pos_x = center_x - im_width / 2
    pos_y = center_y - im_height / 2
    print("coords: ", left, right, top, bottom)
    left, right, top, bottom = int(left), int(right), int(top), int(bottom)
    depth_box = depth_map[left:right, top:bottom]
    pos_z = depth_box.mean()

    return [pos_x, pos_y, pos_z]

def process_frame(frame):
    # Feed camera feed into object detection algorithm to get bounding boxes
    # Show bounding boxes in feed
    yolo_image, out_boxes, out_scores = yolo.detect_image(frame)
    yolo_image = np.array(yolo_image)
    
    #yolo_image.save(dir_output + '/yolo_' + file_img)
                    
    #for i in range(len(out_boxes)): 
    #    top, left, bottom, right = out_boxes[i]
    #    file_detections.write('Banana {} in {}: Top: {}, Left: {}, Bottom: {}, Right: {}, Confidence: {}\n'.format(i+1, file_img, top, left, bottom, right, out_scores[i]))      
    if out_boxes:
        cv2.imshow("Auralizer", yolo_image)
        
        # METHOD 1 - Depth estimation:
        # Feed camera feed into monocular depth estimation algorithm and get depth map
        # Show depth map
        print("shape before depth forward: ", np.array(frame).shape)
        depth_map = depth_model.forward(np.array(frame))
        depth_map = depth_map.squeeze()
        # Upsample the depth map:
        print("depth map shape before upsampling: ", depth_map.shape)
        depth_map = np.array(Image.fromarray(depth_map).resize(yolo_image.shape[:2]))
        print("depth map shape: ", depth_map.shape)
         
        # Combine bounding box and depth to get coordinate of object.

        # METHOD 2 - 
        # Feed camera feed into SLAM and get list of features with coordinates
        # Mark features that can be seen from current frame and that are within bounding box as relating to the object
        # Get coordinates of the feature group that is closest and/or has the highest density of detected features for the sought object

        # METHOD 3 -
        # Calculate center of detected bbox relative to camera center
        # Those coordinates will represent x and y of the current 3D position (with fixed z value)

        # FINAL:
        # Give coordinate of object to auralizer to create sound.
        
        object_position = get_position_bbox(yolo_image, out_boxes, depth_map)
        
        print("Object pos: ", object_position)
        # 0.03 - 0.05 for z position

        audio.set_position(object_position)
        audio.play()
    else:
        cv2.imshow("Auralizer", np.array(frame))


while True:
    frame = cam.get_current_frame()
    if frame is not None:
        process_frame(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
