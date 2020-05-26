import numpy as np
import cv2

from calibration.webcam import Webcam
from object_detection.detect_bananas import YOLO
# Read intrinsic camera parameters, if none detected prompt calibration.
#try:
camera_matrix = np.load("calibration/camera_matrix.npy")
dist_coefs = np.load("calibration/dist_coefs.npy")


# Instantiate all algorithms
cam = Webcam()
yolo = YOLO("object_detection")  # assumes YOLO model weights are downloaded! (see keras yolo readme)

# Create Camera object
# Get camera feed from camera object
cam.start()

while True:
    frame = cam.get_current_frame()
    process_frame(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def process_frame(frame):
    # Feed camera feed into object detection algorithm to get bounding boxes
    # Show bounding boxes in feed
    yolo_image, out_boxes, out_scores = yolo.detect_image(frame)
    
    #yolo_image.save(dir_output + '/yolo_' + file_img)
                    
    #for i in range(len(out_boxes)): 
    #    top, left, bottom, right = out_boxes[i]
    #    file_detections.write('Banana {} in {}: Top: {}, Left: {}, Bottom: {}, Right: {}, Confidence: {}\n'.format(i+1, file_img, top, left, bottom, right, out_scores[i]))
        
        
    cv2.imshow(yolo_image)

    # METHOD 1 - Depth estimation:
    # Feed camera feed into monocular depth estimation algorithm and get depth map
    # Show depth map

    # Combine bounding box and depth to get coordinate of object.

    # METHOD 2 - 
    # Feed camera feed into SLAM and get list of features with coordinates
    # Mark features that can be seen from current frame and that are within bounding box as relating to the object
    # Get coordinates of the feature group that is closest and/or has the highest density of detected features for the sought object

    # FINAL:
    # Give coordinate of object to auralizer to create sound.






