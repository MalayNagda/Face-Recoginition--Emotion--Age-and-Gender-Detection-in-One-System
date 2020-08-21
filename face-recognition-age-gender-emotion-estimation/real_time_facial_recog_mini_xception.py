import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import time
from pathlib import Path
import dlib
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import socket
import matplotlib.pyplot as plt
import os
from os import listdir
import FaceToolKit as ftk
import DetectionToolKit as dtk
import tensorflow as tf

################################################################################################################
verification_threshhold = 0.70
image_size = 160
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1)
v = ftk.Verification()
# Pre-load model for Verification
v.load_model("./osl_data/models/20180408-102900/20180408-102900.pb")
v.initial_input_output_tensors()


d = dtk.Detection()

def cutfaces(image, faces_coord):
    faces = []


    for (x,y,w,h) in faces_coord:
        w_rm = int(0.2*w/2)
        faces.append(image[y : y + h, x + w_rm : x + w - w_rm])
    return faces

def webcam(path):
    PADDING = 25
    face = cv2.CascadeClassifier('./osl_data/models/20180408-102900/haarcascade_frontalface_default.xml')
    webcam = cv2.VideoCapture(0)
    while True:
        ret, img = webcam.read()
        frame = img
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_coord= face.detectMultiScale(gray, 1.2, 7, minSize=(50,50))
        faces = cutfaces(img, faces_coord)
        if (len(faces) != 0):            
            for (x, y, w, h) in faces_coord:
                x1 = x-PADDING
                y1 = y-PADDING
                x2 = x+w+PADDING
                y2 = y+h+PADDING

                img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,255,255),2)
                height, width, channels = frame.shape
                cut_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            cut_image= cv2.resize(cut_image,(image_size ,image_size))
            cv2.imwrite(path, cut_image)   
            break

        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    webcam.release()
    plt.imshow(img)
    cv2.destroyAllWindows()
    return cut_image



def img_to_encoding(img):
    image = plt.imread(img)
    aligned = d.align(image, False)[0]
    return v.img_to_encoding(aligned, image_size)


# data= img_to_encoding("./images/Ali.jpg")


def database_image():
    database = {}
    path='./osl_data/images/'
    try:     
        for f in listdir(path):     
            if f.startswith('.'):
                continue
        # Iterate over index 
            name=''
            for element in range(0, len(f)): 
                if f[element]=='.':
                    break
                name+=f[element]
            #if f is null
            database[name] = img_to_encoding(path+f) 
    except IndexError:
          pass
    return database


def distance(emb1, emb2): 
    diff = np.subtract(emb1, emb2)
    return np.sum(np.square(diff))


def who_is_it(image_path, database):
    flag=0
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding()
    encoding = img_to_encoding(image_path)
    
    
    # Initialize "min_dist" to a large value, say 100 
    min_dist = 1000
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = distance(encoding, db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if min_dist > dist:
            min_dist = dist
            identity = name
  
    if min_dist > verification_threshhold:
        print("Not in the database.")
        flag=1
    else:
        print ("It's " + str(identity) + ", the distance is " + str(min_dist))
        
    if flag==1:
        #image directory
        directory = r'./osl_data/images'
        os.chdir(directory)
        name=input('Please Enter your name Beautiful!!:  ')
        cv2.imwrite(name+'.jpg',img0)
        print('Hi '+name)
        id=name;
    else:
        id=identity;
    return id
img0=webcam('./osl_data/Verify/Test.jpg')
data=database_image()
name1=who_is_it('./osl_data/Verify/Test.jpg', data)
print(name1)
################################################################################################

time.sleep(3)
print('Wait for 3 sec')

sock =  socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('10.157.149.122', 10000)


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
               font_scale=0.45):
    x, y = point
    cv2.putText(image, label, point, font, font_scale, (0, 0, 255), lineType=cv2.LINE_AA)

# parameters for loading data and images
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = '_mini_XCEPTION.48-0.62.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]


# starting video streaming

cv2.namedWindow('your_face')
count = 0;
camera = cv2.VideoCapture(0)
while True:
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
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)
    frame = camera.read()[1]
    #reading the frame
    frame1=frame;
    frame = imutils.resize(frame,width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    time.sleep(1)
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
            # the ROI for classification via the CNNqq
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
            (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 2)
            cv2.putText(frameClone, label, (fX+3, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                          (0, 0, 255), 2)


    detected = detector(frame1, 1)
    img_h, img_w, _ = np.shape(frame1)
    faces1 = np.empty((len(detected), img_size, img_size, 3))
    if len(detected) > 0:
        for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(frame1, (x1, y1), (x2, y2), (255, 0, 0), 2)
                faces1[i, :, :, :] = cv2.resize(frame1[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

        results = model.predict(faces1)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
        for i, d in enumerate(detected):
                label1 = "{}, {}".format(int(predicted_ages[i]),
                                        "M" if predicted_genders[i][0] < 0.5 else "F")
                #print(label1)
                draw_label(frameClone, (fX+3, fY+14), label1)
    count+=1;
    if count >= 2:
            message = "The detected emotion, age, gender and name is: " + label+', '+ label1+', '+name1
            sent = sock.sendto(message.encode('utf-8'), server_address)
            count = 0
            print('sent');
            print(message)
    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()