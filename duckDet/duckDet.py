# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 22:02:47 2020
author: _ConneR_
email: crisprhhx@qq.com
version: 1.1.0
"""
import imutils
from imutils.video import FPS
import time
from utils.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2

pbpath='../data/frozen_inference_graph.pb'
pbtxtpath='../data/graph.pbtxt'
cfg = '../data/yolov3-obj.cfg'
weight =  '../data/yolov3-obj_final.weights'
testpath='../img/test1.jpg'
img_path='../imgs/map_proj.jpg'
map_path='../imgs/map.png'
ct = CentroidTracker()

#####################################################################
#####################################################################
############This part of the code needs to be more tidy##############
#####################################################################
#####################################################################

#Failed to use callback func, so the points are manually specified.
pts = np.array([[1231, 171], [1231, 716], [154, 144], [9, 688]],dtype = "float32")

cv2.namedWindow('map',cv2.WINDOW_NORMAL)
mapimg=cv2.imread(map_path)
'''
print("[INFO] assign four points in format:[upleft,upright,downleft,downright]")
def draw_circle(event,x,y,flags,param): #鼠标事件回调
    global pts
    if event==cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),10,(255,0,0),-1)
        pts.append([x,y])
        print('[INFO]You have assigned:',len(pts),'/ 4 points.')
img = cv2.imread(img_path)  #此处改为自己文件位置
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    if len(pts) == 4:
        break
    if cv2.waitKey(20)&0xFF=='q':  #按ESC退出
        break
cv2.destroyAllWindows()
print('Points got:\n',pts,'\n')
pts = np.array(pts,dtype ='float32').reshape(-1,1,2)
'''
point1 = pts
point2 = np.array([[28, 0], [450, 2], [30, 371], [452, 374]],dtype = "float32")
M = cv2.getPerspectiveTransform(point1,point2)



#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################



#<<--main-->>#

print("[INFO] loading model...")
#>>>using tensorflow ssd based pd and pdtxt file:
#net = cv2.dnn.readNetFromTensorflow(pbpath, pbtxtpath)
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#>>>

#>>>using yolo darknet based weight and cfg file:
net = cv2.dnn.readNetFromDarknet(cfg, weight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layers_names = net.getLayerNames()
output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
#>>>

print("[INFO] starting video stream...")
cap = cv2.VideoCapture('D:/darknet-master/build/darknet/x64/data/rmSJTU.mp4')
fps = FPS().start()
while True:
	# read the next frame from the video stream and resize it
	ret,frame = cap.read()
	mapimg_copy = mapimg.copy()
	rows = frame.shape[0]
	cols = frame.shape[1]
	#>>>using tensorflow ssd based pd and pdtxt file:
	#net.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
	#outputs = net.forward()
	#>>>
	
	#>>>using yolo darknet based weight and cfg file:
	#if fps too slow,decrease size,but should be devisible by 32, like 160,192,288,416,etc.
	net.setInput(cv2.dnn.blobFromImage(frame,0.00392, size=(128, 128), swapRB=True, crop=False))
	outputs = net.forward(output_layers)
	#>>>
	
	(H, W) = frame.shape[:2]
	rects = []
	'''
	#Loop For Tensorflow SSD Pb Model:
	for detection in outputs[0,0,:,:]:
		score = float(detection[2])
		if score > 0.3:
			left = detection[3] * cols
			top = detection[4] * rows
			right = detection[5] * cols
			bottom = detection[6] * rows
			cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
			box = detection[3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))
	'''	
	#Loop For Yolov3 Model:
	#'''
	for output in outputs:
		for detect in output:
			score = detect[5:]			
			if score > 0.38:
				center_x = int(detect[0] * W)
				center_y = int(detect[1] * H)
				w = int(detect[2] * W)
				h = int(detect[3] * H)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				#cv2.rectangle(frame, (x,y),(x+w,y+h), (23, 230, 210), thickness=2)
				box = [x,y,x+w,y+h] * np.array([1, 1, 1, 1])
				rects.append(box.astype("int"))
	
	#'''
	objects = ct.update(rects)
	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)		
		cv2.rectangle(frame,(centroid[0]-8, centroid[1]-8),(centroid[0]+8, centroid[1]+8),(23, 230, 210), thickness=2)
		
		#Affine transformation, mapping the target to the map
		obj_pts = np.float32([centroid[0], centroid[1]]).reshape(-1,1,2)
		out_pts = cv2.perspectiveTransform(obj_pts, M)
		point=np.squeeze(out_pts[0],0)
		point = tuple(point)
		#assert (False),"debug"
		cv2.circle(mapimg_copy,point,5,(0,0,255),-1)
		
	cv2.imshow('map',mapimg_copy)
	key = cv2.waitKey(1)	
	fps.update()	
	#frame = imutils.resize(frame, height=500, width=890)
	cv2.imshow('frame', frame)
	key = cv2.waitKey(1)
	
	if key == ord("q"):
		break
	#time.sleep(0.5)

# do a bit of cleanup
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
















	
