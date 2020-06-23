import glob, os
from math import trunc

dir_labels = '/data_b/YCB_Video_Dataset/data/'
out_file = 'ycb_yolo_annotations.txt'


def files(dir):
#    for root, dirs, files in os.walk(dir):
#        for file in files:
#    	    if file.endswith('.txt'):
#        		yield os.path.join(root, file)

    os.chdir(dir_labels)
    for file in glob.glob('*.txt', recursive=True):
        yield file


file_generator = files(dir_labels)

with open(out_file, 'w') as f_out:

    for file in file_generator:

        with open(file) as f_in:
            objects = f_in.readlines()
            objects = [x.strip() for x in objects]

            img_filename = file.split('-')[0] + '-color.png'

            f_out.write(img_filename)
            print(img_filename)

            for obj in objects:
                obj = obj.split()
                class_label = obj[0].split('_')[0]
                class_label = class_label.lstrip('0')
                bbox = obj[1:]
                x_min = trunc(float(bbox[0]))
                y_min = trunc(float(bbox[1]))
                x_max = trunc(float(bbox[2]))
                y_max = trunc(float(bbox[3]))

                converted_string = ' {},{},{},{},{}'.format(x_min, y_min, x_max, y_max, class_label)

                f_out.write(converted_string)

            f_out.write('\n')
