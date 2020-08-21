
# 03_FaceRecognition-verification&Identification
# by abdul and Amith
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir

import FaceToolKit as ftk
import DetectionToolKit as dtk



verification_threshhold = 0.70
image_size = 160
v = ftk.Verification()
# Pre-load model for Verification
v.load_model("./models/20180408-102900/20180408-102900.pb")
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
    face = cv2.CascadeClassifier('./models/20180408-102900/haarcascade_frontalface_default.xml')
    webcam = cv2.VideoCapture(0)
    while True:
        ret, img = webcam.read()
        frame = img
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_coord= face.detectMultiScale(gray, 1.2, 7, minSize=(50,50))
        faces = cutfaces(img, faces_coord)
        
        
        
        if (len(faces) != 0):
            
            
            #cv2.imwrite('img_test.jpg',faces[0])
            
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
    path='./images/'
    try:     
        for f in listdir(path):     
            if f.startswith('.'):
                continue
           # print(f)
        # Iterate over index 
            name=''
            for element in range(0, len(f)): 
                if f[element]=='.':
                    break
                name+=f[element]
           # print(name)
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
        directory = r'./images'
        os.chdir(directory)
        name=input('Please Enter your name Beautiful!!:  ')
        cv2.imwrite(name+'.jpg',img0)
        print('Hi '+name)
        
img0=webcam('./Verify/Test.jpg')
data=database_image()
who_is_it('./Verify/Test.jpg', data)