#!/usr/bin/python3

import numpy as np

def estimate(camera_matrix, depth_image, bounding_box):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    depth_scaling_factor = 10000.0 #FIXME: output from monocular depth estimation?

    height = depth_image.shape[0]
    width = depth_image.shape[1]
    xv = np.arange(width)
    yv = np.arange(height)
    X, Y = np.meshgrid(xv, yv)

    x = ( (X - cx) * dep / fx )
    y = ( (Y - cy) * dep / fy )
    xyz = np.stack([x,y,dep], axis=2)

    box_x, box_y, box_w, box_h = bounding_box
    
    top, left, bottom, right = bounding_box

    est_x = np.median(xyz[top:bottom, left:right, 0])
    est_y = np.median(xyz[top:bottom, left:right, 1])
    est_z = np.median(xyz[top:bottom, left:right, 2])

    return est_x, est_y, est_z
