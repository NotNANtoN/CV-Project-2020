cd object_detection/keras_yolo3
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5

cd Densedepth
wget https://s3-eu-west-1.amazonaws.com/densedepth/nyu.h5
