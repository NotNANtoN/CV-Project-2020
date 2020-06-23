lines = [line for line in open('ycb_yolo_annotations.txt') if '11 ' in line]

new_lines = []

for line in lines:
    words = line.split()
    new_line = words[0]
    for obj in words[1:]:
        if obj.endswith('11'):
            new_line += ' ' + obj[:-2] + '0'
    new_line += '\n'
    new_lines.append(new_line)


open('ycb_yolo_annotations_bananas_02.txt', 'w').writelines(new_lines)
