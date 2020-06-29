# CV-Project-2020

## Setup:
First install the required packages using `python -m pip install -r requirements.txt`.

Now you need to download the weights of the pretrained networks using `bash get_yolo_and_depth_weights.sh`.

Finally, you can call the `auralize.py` script, using either `python auralize.py -s cam` to auralize the location of 
objects seen from your webcam (currently only supports bananas), or you can call it simply as `python auralize.py` and
by default the system will choose a video from the YCB dataset. By passing a path to the `-s` argument you can also 
test auralization of a different video.

To build the OrbSLam2 bindings first make sure you have all requirements, mentionend in the ORB_SLAM2/README, file installed. Then switch to the ./orbslam_pybindings directory and execute the bash script make_pybindings.sh script. This should now build the whole C-Project as well as our bindings. The python module file is located in orbslam_pybindings/lib/ and can be imported by appending this path to the system path and then calling 'import orbslam2'. 
There is an additional test_tum.py file in the directory which you can use to test the bindings. You need to download the video files from https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz and unpack this directory into the orbslam_pybindings/examples/ directory.

## General system
### Overview:
We have these main sub-systems to finally produce a 3D sound based on an image stream:
1. Object detection - using Yolov3 or Mask-RCNN to detect the bounding box of the object.
2. Depth estimation - use SLAM or pretrained monocular depth estimation in combination with bounding box to find 3D location of the object.
3. Sound creation - given x,y,z coordinate play a 3D sound relative to the current position of the camera
### How to use:
Run the `auralize.py` script. It will take the camera feed as an input and show as an input the observed image with marked objects, along with a stereo sound signal indicating the distance to the object.
Weights for detection model (save to model_data/):
large: (~240MB) https://drive.google.com/open?id=1WD8g2u_MI7KOpZG3bGm9yANEEPVkZhi0
tiny: (~35MB) https://drive.google.com/open?id=1jBFaKNq9ITBdvnDtuQe58DgF5B68Gj6F
### System evaluation:
1. Speed: The faster the better. We need to evaluate how many FPS our system is required to be able to process to be useful.
2. Accuracy: The more accurate the better if the system is still fast enough. Also here we need to find a way of how much inprecision in the x,y domain and in the depth domain is acceptable to still handle the target objects.
3. Requirements: The fewer the better. Ideally this system should be useable with a monocular RGB camera and stereo headphones only.
### Data
We decided on the YCB video dataset. (TODO: which object classes? All of them?)
### Camera calibration
To localize an object more precisely it would help to calibrate the camera before use. For this we need to write a script still.

## Object detection:
### Overview:
Given an RGB image (and potentially an object class) returns either all bounding boxes or only the bounding boxes of the given object class.
### System evaluation:
We need to evaluate inference speed and classification metrics. For classification it matters if the correct objects is detected and how precisely it is detected. The former is more important than the latter as the precise location might self-correct in later video frames.
### Issues:
We need to decide:
  1. Yolov3 (speed) vs Mask-RCNN (accuracy)
  2. Which pretrained weights to use (what dataset was the model trained on, which classes?)
  3. Whether we fine-tune on YCB data or not.
  
## Depth estimation:
### Overview:
To estimate depth we can either use monocular depth estimation or try to build 3D map of all detected features using SLAM.
### Monocular Depth Estimation
placeholder
### SLAM
After getting it to compile, we need to check whether our YCB test objects

## Sound creation:
### Overview:
Given a x,y,z values for the distance to the center of an object, return

