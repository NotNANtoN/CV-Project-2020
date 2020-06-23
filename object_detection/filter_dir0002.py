open('ycb_yolo_annotations_bananas_train.txt', 'w').writelines(line for line in open('ycb_yolo_annotations_bananas.txt') if not '/0002/' in line)
