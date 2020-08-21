# -*- coding: utf-8 -*-
# 02_FaceDetection&Alignment-MTCNN.ipynb

import tensorflow as tf
import numpy as np
import cv2
from detection.mtcnn import detect_face


default_color = (0, 255, 0) #BGR
default_thickness = 2


with tf.Graph().as_default():
    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)


minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor


import matplotlib.pyplot as plt
img = plt.imread("./images/khandevane.jpg")
plt.imshow(img)
plt.show()


bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)


for bounding_box in bounding_boxes:
        pts = bounding_box[:4].astype(np.int32)
        pt1 = (pts[0], pts[1])
        pt2 = (pts[2], pts[3])
        cv2.rectangle(img, pt1, pt2, color=default_color, thickness=default_thickness)

plt.imshow(img)
plt.show()


for i in range(points.shape[1]):
        pts = points[:, i].astype(np.int32)
        for j in range(pts.size // 2):
            pt = (pts[j], pts[5 + j])
            cv2.circle(img, center=pt, radius=1, color=default_color, thickness=default_thickness)

plt.imshow(img)
plt.show()

