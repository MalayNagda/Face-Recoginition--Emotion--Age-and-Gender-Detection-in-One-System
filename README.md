# Face Recoginition, Emotion, Age and Gender Detection in One System
 

<p align="center">
  <img src="images/vid_final_sys.gif">
  <p align="center">Final system in action</p>
</p>

The objective of the project was to have face recoginition, emotion detection, age and gender estimation in one system. We first tested the three systems individually and then went forward to combine them. 

## Flow Diagram of the Running System
<p align="center">
  <img src="images/system_flowchart.png">
  <p align="center"> Overview of how the entire system works.</p>
</p>

## Dependencies
* Python 3.7.x

Tested on
* Windows 10 Home, Python 3.7.5, TensorFlow 2.1.0

Initially the project was implemented on TensorFlow 1.x. I have updated the code to be compatible with TensorFlow 2.0. 

## Usage
The code to run the system can be found in 'real_time_facial_recog_mini_xception.py'. 

We have implemented face recoginition using One shot Learning and the related resources as provided in 'Convolutional Neural Networks' course by Deeplearning.ai on Coursera.
For emotion detection, [mini-Xception model](https://arxiv.org/pdf/1710.07557.pdf) was trained but with datasets different from that mentioned in the paper.

We used a combination of 4 datasets for this purpose, of which two links are broken now. The other two are-
* [The Japanese Female Facial Expression (JAFFE) Database](https://zenodo.org/record/3451524#.X0AJy8hKiUk)
* [FacesDB](http://app.visgraf.impa.br/database/faces/)

We used [this repository](https://github.com/yu4u/age-gender-estimation) for integrating age and gender estimation within our project. 

<p align="center">
  <img src="images/face_recog.png">
  <p align="center">One-shot Learning. a) Face of the person captured. b) Person is not in the system. c) Person stored in the database. d) Person re-appears in front of the camera and
the AI agent recognizes the person
.</p>
</p>

<p align="center">
  <img src="images/final_sys.png">
<p align="center">Emotion, Age and Gender Detection</p>
</p>

<p align="center">
  <img src="images/UDP.png">
  <p align="center">UDP Connection Result with another system</p>
</p>

## Authors
* Abdul Mohd Wahab- face recognition and emotion detection models  
* Amith Lawrence- face recognition and emotion detection models
* Anusha Vaidya- age and gender estimation model, UDP communication
* Malay Nagda- age and gender estimation model, UDP communication and combining the three systems
* Shivani Shah- literature review, code debugging, protyping and final face recognition and emotion detection models

## Acknowledgements
Dr. Suren Jayasuriya and Mr. Christian Ziegler were are mentors for the project and gave valuable guidance in terms of the expectation of how the final system should look like.
