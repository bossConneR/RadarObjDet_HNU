# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 20:39:00 2020

@author: 10720
"""

import imutils
import cv2 as cv
pbpath='../data/frozen_inference_graph.pb'
pbtxtpath='../data/graph.pbtxt'
testpath='../imgs/test3.jpg'

cvNet = cv.dnn.readNetFromTensorflow(pbpath, pbtxtpath)

img = cv.imread(testpath)
rows = img.shape[0]
cols = img.shape[1]
cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
cvOut = cvNet.forward()

for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.3:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

img = imutils.resize(img, height=500, width=890)
cv.imshow('TensorFlow MobileNet-SSD', img)
cv.waitKey(0)