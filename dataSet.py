# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:11:49 2020

@author: Bibek77
"""


import cv2 
import numpy as np
import sys

cpt=0

vidStream=cv2.VideoCapture(0)

while True:
    ret, frame= vidStream.read()
    cv2.imshow("test frame", frame)
    
    cv2.imwrite(r"D:\FaceRecognitionLBPH\Image\1\image%04i.jpg" %cpt,frame)
    cpt +=1
    
    if cv2.waitKey(10)==ord('q'):
        break
        
    