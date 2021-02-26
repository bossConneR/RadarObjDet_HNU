#!/usr/bin/python
#!--*-- coding:utf-8 --*--
import cv2
import matplotlib.pyplot as plt


pb_file = '../data/frozen_inference_graph_ssd.pb'
pbtxt_file = '../data/graph_ssd.pbtxt'
net = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
labelmap = {0:'car_blue',1:'car_red',2:'car_unknown',3:'armor_blue',4:'armor_red',5:'armor_blue'}

score_threshold = 0.5

img_file = '../imgs/patch1.jpg'

img_cv2 = cv2.imread(img_file)
height, width, _ = img_cv2.shape
net.setInput(cv2.dnn.blobFromImage(img_cv2, 
                                   size=(300, 300), 
                                   swapRB=True, 
                                   crop=False))

out = net.forward()
#print(out)

for detection in out[0, 0, :,:]:
    
    score = float(detection[2])
    if score > score_threshold:
        print(detection)
        print(labelmap[detection[1]])
        left = detection[3] * width
        top = detection[4] * height
        right = detection[5] * width
        bottom = detection[6] * height
        cv2.rectangle(img_cv2, 
                      (int(left), int(top)), 
                      (int(right), int(bottom)), 
                      (23, 230, 210), 
                      thickness=2)

t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % \
            (t * 1000.0 / cv2.getTickFrequency())
cv2.putText(img_cv2, label, (0, 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

plt.figure(figsize=(10, 8))
plt.imshow(img_cv2[:, :, ::-1])
plt.show()