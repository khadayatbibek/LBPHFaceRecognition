# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:51:45 2020

@author: Bibek77
"""


import numpy as np
import cv2
import os

import face_recognition as fr
print (fr)

test_img=cv2.imread(r'D:\FaceRecognitionLBPH\Test_img.jpg')      #Give path to the image which you want to test


faces_detected,gray_img=fr.faceDetection(test_img)
print("face Detected: ",faces_detected)


face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'D:\FaceRecognitionLBPH\trainedData.yml')  #Give path of where trainingData.yml is saved

name={0:"Bibek",1:"2nd img bibek"}              #Change names accordingly.  If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1. 

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print ("Confidence :",confidence)
    print("label :",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))

cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows