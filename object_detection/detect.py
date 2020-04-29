from detect_bananas import YOLO
from PIL import Image
import os

yolo = YOLO()

dir_input='input/'
dir_output='output/'

with open(dir_output + 'yolo_detections.txt', 'w') as file_detections:
    for file_img in os.listdir(dir_input):
        if file_img.endswith('.png'):
            try:
                image = Image.open(dir_input + file_img)
            except:
                print('Error opening ' + file)
                break
            else:
                yolo_image, out_boxes, out_scores = yolo.detect_image(image)
                yolo_image.save(dir_output + '/yolo_' + file_img)
                
                for i in range(len(out_boxes)): 
                    top, left, bottom, right = out_boxes[i]
                    file_detections.write('Banana {} in {}: Top: {}, Left: {}, Bottom: {}, Right: {}, Confidence: {}\n'.format(i+1, file_img, top, left, bottom, right, out_scores[i]))

yolo.close_session()