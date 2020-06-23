import glob, os

dir_labels = 'labels'

im_width = 640
im_height = 480

def files(dir):
    os.chdir(dir_labels)
    for file in glob.glob('*.txt'):
        yield file

file_generator = files(dir_labels)

for file in file_generator:

    with open(file) as f_in:
        objects = f_in.readlines()
        objects = [x.strip() for x in objects]
        
        new_filename = file.split('-')[0] + '.txt'
        with open(new_filename, 'w') as f_out:

            for obj in objects:
                obj = obj.split()
                class_label = obj[0].split('_')[0]
                class_label = class_label.lstrip('0')
                bbox = obj[1:]
                x_center = (float(bbox[0]) + float(bbox[2])) / 2.0 / im_width
                y_center = (float(bbox[1]) + float(bbox[3])) / 2.0 / im_height
                width = (float(bbox[2]) - float(bbox[0])) / im_width
                height = (float(bbox[3]) - float(bbox[1])) / im_height

                converted_string = '{} {} {} {} {}'.format(class_label, x_center, y_center, width, height)
                
                f_out.write(converted_string + '\n')