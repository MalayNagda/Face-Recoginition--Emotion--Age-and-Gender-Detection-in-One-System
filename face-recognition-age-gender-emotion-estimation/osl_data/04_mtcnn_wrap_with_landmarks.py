# -*- coding: utf-8 -*-
# 04_MTCNN-wrap-with-landmarks.ipynb

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
img = plt.imread("./images/Mohd.jpg")
plt.imshow(img)
plt.show()


bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)


for bounding_box in bounding_boxes:
        pts = bounding_box[:4].astype(np.int32)
        pt1 = (pts[0], pts[1])
        pt2 = (pts[2], pts[3])
        cv2.rectangle(img, pt1, pt2, color=default_color, thickness=default_thickness)


for i in range(points.shape[1]):
        pts = points[:, i].astype(np.int32)
        for j in range(pts.size // 2):
            pt = (pts[j], pts[5 + j])
            cv2.circle(img, center=pt, radius=1, color=default_color, thickness=default_thickness)

plt.imshow(img)
plt.show()

from skimage import transform as trans

img = plt.imread("./images/Mohd.jpg")
src = np.array([
       [  54.70657349,   73.85186005],
       [ 105.04542542,   73.57342529],
       [  80.03600311,  102.48085785],
       [  59.35614395,  131.95071411],
       [ 101.04272461,  131.72013855]], dtype=np.float32)

landmark = points[:, 0].reshape( (2,5) ).T

dst = landmark.astype(np.float32)
tform = trans.SimilarityTransform()
tform.estimate(dst, src)
M = tform.params[0:2,:]

M

wrapped = cv2.warpAffine(img,M,(160,160), borderValue = 0.0)

plt.imshow(wrapped)
plt.show()
cv2.imwrite("./images/Mohd.jpg", wrapped[:,:,-1::-1])
