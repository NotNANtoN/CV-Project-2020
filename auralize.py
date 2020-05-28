import numpy as np
import cv2
from PIL import Image

from calibration.webcam import Webcam
from object_detection.detect_bananas import YOLO
from audio_playground.Audio import Audio
from depth2coords import depth2coords


def process_frame(frame):
    # Feed camera feed into object detection algorithm to get bounding boxes
    # Show bounding boxes in feed
    yolo_image, out_boxes, out_scores = yolo.detect_image(frame)
    yolo_image = np.array(yolo_image)
    
    #yolo_image.save(dir_output + '/yolo_' + file_img)
                    
    #for i in range(len(out_boxes)): 
    #    top, left, bottom, right = out_boxes[i]
    #    file_detections.write('Banana {} in {}: Top: {}, Left: {}, Bottom: {}, Right: {}, Confidence: {}\n'.format(i+1, file_img, top, left, bottom, right, out_scores[i]))
        
    
    cv2.imshow("Yolo processed image", yolo_image)

    # METHOD 1 - Depth estimation:
    # Feed camera feed into monocular depth estimation algorithm and get depth map
    # Show depth map

    # Combine bounding box and depth to get coordinate of object.
    #for bounding_box in out_boxes:
    #    est_x, est_y, est_z = depth2coords.estimate(camera_matrix, depth_image, bounding_box)
    #    print("Image: {}\tx {}\ty {}\tz {}".format(depth_image, est_x, est_y, est_z))

    # METHOD 2 - 
    # Feed camera feed into SLAM and get list of features with coordinates
    # Mark features that can be seen from current frame and that are within bounding box as relating to the object
    # Get coordinates of the feature group that is closest and/or has the highest density of detected features for the sought object

    # METHOD 3 -
    # Calculate center of detected bbox relative to camera center
    # Those coordinates will represent x and y of the current 3D position (with fixed z value)


    # FINAL:
    # Give coordinate of object to auralizer to create sound.

    if out_boxes:
        object_position = get_position_bbox(yolo_image, out_boxes)

        audio.play()
        audio.set_position(object_position)

def get_position_bbox(img, out_boxes):
    top, left, bottom, right = out_boxes[0]
    center_x = (right - left) / 2 + left
    center_y = (bottom - top) / 2 + top

    im_width, im_height, _ = img.shape

    pos_x = center_x - im_width / 2
    pos_y = center_y - im_height / 2
    pos_z = 1

    return [pos_x, pos_y, pos_z]
    

# Read intrinsic camera parameters, if none detected prompt calibration.
#try:
camera_matrix = np.load("calibration/camera_matrix.npy")
dist_coefs = np.load("calibration/dist_coefs.npy")


# Instantiate all algorithms
cam = Webcam()
yolo = YOLO("object_detection")  # assumes YOLO model weights are downloaded! (see keras yolo readme)
audio = Audio("audio_playground/sound.wav")

# Create Camera object
# Get camera feed from camera object
cam.start()

while True:
    frame = cam.get_current_frame()
    if frame is not None:
        #print(type(frame))
        #print(frame.shape)
        frame = Image.fromarray(frame, 'RGB')
        #frame = np.array(frame)
        process_frame(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break








