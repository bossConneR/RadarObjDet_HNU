# -*- coding: utf-8 -*-
from imutils.video import FPS
from utils.centroidtracker import CentroidTracker
import numpy as np
import cv2

map_path='../imgs/map.png'
pb_file = '../data/frozen_inference_graph_ssd.pb'
pbtxt_file = '../data/graph_ssd.pbtxt'
video_path = 'D:/darknet-master/build/darknet/x64/data/rmSJTU.mp4'
#Instantiate two trackers
ct_r = CentroidTracker()
ct_b = CentroidTracker()

score_threshold = 0.5
carblue=(255,144,30)
red = (48,48,255)
bigred=(0,0,255)
blue = (255,0,0)
armorblue=(205,0,0)
carred=(106,106,255)
armorred=(0,69,255)
grey = (128,128,128)
originyellow=(23, 230, 210)
labelmap = {0:'car_blue',1:'car_red',2:'car_unknown',3:'armor_blue',4:'armor_red'}
#the points are manually specified.
point1 = np.array([[1231, 171], [1231, 716], [154, 144], [9, 688]],dtype = "float32")
point2 = np.array([[28, 0], [450, 2], [30, 371], [452, 374]],dtype = "float32")
cv2.namedWindow('map',cv2.WINDOW_NORMAL)
mapimg=cv2.imread(map_path)

#get perspective transform by using two points.
M = cv2.getPerspectiveTransform(point1,point2)

def color_assign(idx):
    #There may be errors in the match between idx and color/label.
    idx = (int)(idx)
    car_obj=2
    if (idx == 0 or idx == 2):
        color = carred
        car_obj = 0
    elif (idx == 1):
        color = carblue
        car_obj = 1
    elif (idx == 4):
        color = armorblue
    elif (idx == 3 or idx == 5):
        color = armorred
    else:
        color = grey
    return color, car_obj
        
def load_net_from_tfmodel(pb_file, pbtxt_file):
    #load opencv nets from tensorflow model.
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net

def track_vis(objects, frame, mapimg_copy, color):
    #visualize the tracking of objects.
    for (objectID, centroid) in objects.items():
        if(color == 'blue'):
            text = "BLUE {}".format(objectID)
            color_ = blue
        elif(color == 'red'):
            text = "RED {}".format(objectID)
            color_ = red
        else:
            assert False, "param 'color' should be blue or red."
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4,carred, -1)       #BGR
        cv2.rectangle(frame,(centroid[0]-8, centroid[1]-8),(centroid[0]+8, centroid[1]+8),color_, thickness=2)
        obj_pts = np.float32([centroid[0], centroid[1]]).reshape(-1,1,2)
        out_pts = cv2.perspectiveTransform(obj_pts, M)
        point=np.squeeze(out_pts[0],0)
        point = tuple(point)
        distance = pow((pow((point[0]-238),2) + pow((point[1]-325),2)),0.5)
        point_arr=np.array(point,dtype=int)
        cv2.circle(mapimg_copy,point,5,color_,-1)
        if(color == 'red'):
            if(distance < 150):
                cv2.rectangle(mapimg_copy,(point_arr[0]-15, point_arr[1]-15),(point_arr[0]+15, point_arr[1]+15),bigred, thickness=2)
                cv2.putText(mapimg_copy,'WARNING!', (point_arr[0] - 15, point_arr[1] - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, bigred, 2)
            else:
                cv2.putText(mapimg_copy, text, (point_arr[0] - 10, point_arr[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, carred, 2)
        elif(color == 'blue'):
            cv2.circle(mapimg_copy,(238,325),5,(0, 255, 0),-1)
            cv2.putText(mapimg_copy, text, (point_arr[0] - 10, point_arr[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, carblue, 2)
            cv2.putText(mapimg_copy, 'BASE', (238,325),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

def main(video_path):
    net = load_net_from_tfmodel(pb_file, pbtxt_file)
    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(video_path)
    fps = FPS().start()
    while True:
        # read the next frame from the video stream and resize it
        ret,frame = cap.read()
        mapimg_copy = mapimg.copy()
        rows = frame.shape[0]
        cols = frame.shape[1]
        #>>>using tensorflow ssd based pd and pdtxt file:
        net.setInput(cv2.dnn.blobFromImage(frame,size=(300, 300),swapRB=True,crop=False))
        outputs = net.forward()
        #>>>
        
        (H, W) = frame.shape[:2]
        rects_r = []
        rects_b = []
        #Loop For Tensorflow SSD Pb Model:
        for detection in outputs[0,0,:,:]:
            score = float(detection[2])
            if score > 0.6: 
                #print(detection)
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                idx = detection[1]
                color, car_obj = color_assign(idx)
                #if (color == grey):print(idx)
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), color, thickness=2)
                box = detection[3:7] * np.array([W, H, W, H])
                if(car_obj == 0 ):
                    rects_r.append(box.astype("int"))
                elif(car_obj == 1):
                    rects_b.append(box.astype("int"))

        objects_r = ct_r.update(rects_r)
        objects_b = ct_b.update(rects_b)
        # loop over the tracked objects
        track_vis(objects_r, frame, mapimg_copy, 'red')
        track_vis(objects_b, frame, mapimg_copy, 'blue')
        
        cv2.imshow('map',mapimg_copy)
        key = cv2.waitKey(1)    
        fps.update()    
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        
        if key == ord("q"):
            break
    
    # do a bit of cleanup
    fps.qstop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(video_path)













    
