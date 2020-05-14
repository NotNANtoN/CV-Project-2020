#!/usr/bin/python3

import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#camera intrinsic parameters
fx = 1066.8
fy = 1067.5
cx = 313.0
cy = 241.3
depth_scaling_factor = 10000.0

rgb = imageio.imread('images/000001-color.png')
dep = imageio.imread('images/000001-depth.png') / depth_scaling_factor

height = dep.shape[0]
width = dep.shape[1]
xv = np.arange(width)
yv = np.arange(height)
X, Y = np.meshgrid(xv, yv)

x = ( (X - cx) * dep / fx )
y = ( (Y - cy) * dep / fy )
xyz = np.stack([x,y,dep], axis=2)

#cup bounding box
#box_x, box_y, box_w, box_h = 330, 100, 170, 190

#box bounding box
box_x, box_y, box_w, box_h = 180, 120, 155, 200

bbox1 = patches.Rectangle((box_x, box_y),box_w,box_h,linewidth=1,edgecolor='r',facecolor='none')
bbox2 = patches.Rectangle((box_x, box_y),box_w,box_h,linewidth=1,edgecolor='r',facecolor='none')
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(rgb)
ax1.add_patch(bbox1)
ax2.imshow(dep)
ax2.add_patch(bbox2)
plt.show()

est_x = np.median(xyz[box_y:box_y+box_h, box_x:box_x+box_w, 0])
est_y = np.median(xyz[box_y:box_y+box_h, box_x:box_x+box_w, 1])
est_z = np.median(xyz[box_y:box_y+box_h, box_x:box_x+box_w, 2])

print("Estimated coordinates: x={}, y={}, z={}".format(est_x, est_y, est_z))
