# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 23:55:53 2019

@author: mnagd
"""

from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import time
from keras.preprocessing.image import img_to_array

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


#@contextmanager
#def video_capture(*args, **kwargs):
#    cap = cv2.VideoCapture(*args, **kwargs)
#   try:
#        yield cap
#    finally:
#        cap.release()

#def yield_images():
#    # capture video
#    with video_capture(0) as cap:
#        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
#        while True:
#            # get video frame
#            ret, img = cap.read()
#
#            if not ret:
#                raise RuntimeError("Failed to capture image")
#
#           yield img


#def yield_images_from_dir(image_dir):
#    image_dir = Path(image_dir)

#    for image_path in image_dir.glob("*.*"):
#        img = cv2.imread(str(image_path), 1)

#        if img is not None:
#            h, w, _ = img.shape
#            r = 640 / max(w, h)
#            yield cv2.resize(img, (int(w * r), int(h * r)))'''


#def main():
cap = cv2.VideoCapture(0)
args = get_args()
depth = args.depth
k = args.width
weight_file = args.weight_file
margin = args.margin
image_dir = args.image_dir

if not weight_file:
    weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                            file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    # for face detection
    #detector = dlib.get_frontal_face_detector()
detection_model_path = 'C:/Users/mnagd/Desktop/Sem 1/haarcascade_frontalface_default.xml'
    # load model and weights

while True:
    
    ret, img = cap.read()
    #image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)
    #for img in image_generator:
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 #       img_h, img_w = np.shape(input_img)

        # detect faces using dlib detector
        #detected = detector(input_img, 1)
    detector = cv2.CascadeClassifier(detection_model_path)
    faces = detector.detectMultiScale(input_img,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        #faces = np.empty((len(detected), img_size, img_size, 3))
    time.sleep(1)
    predicted_genders=0.0
    predicted_ages=0.0
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
            # the ROI for classification via the CNNqq
        roi = img[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        results = model.predict(results)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
        #label = format(int(predicted_ages),
    
        for i,d in enumerate(faces):
            #label = "{}, {}".format(int(predicted_ages[i]),
            #                            "M" if predicted_genders[i][0] < 0.5 else "F")
            if predicted_genders[i][0] < 0.5:
                print("M")
            else: 
                print("F")
        #print(predicted_genders[0])
            print(predicted_ages[i])

    cv2.imshow("result", img)
        #key = cv2.waitKey(-1) if image_dir else cv2.waitKey(3)

        #if key == 27:  # ESC
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

'''        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))'''

            # predict ages and genders of the detected faces
        
            # draw results
           # for i, d in enumerate(detected):
           #     label = "{}, {}".format(int(predicted_ages[i]),
           #                             "M" if predicted_genders[i][0] < 0.5 else "F")
           #     draw_label(img, (d.left(), d.top()), label)

       


#if __name__ == '__main__':
#    cap = cv2.VideoCapture(0)
#    main()
