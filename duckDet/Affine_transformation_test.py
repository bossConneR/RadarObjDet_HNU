import cv2
import numpy as np

img_path='../imgs/map_proj.jpg'
map_path='../imgs/map.png'
obj_pts = [[858, 342], [1140, 461], [618, 575], [265, 264]]
obj_pts = np.float32(obj_pts).reshape(-1,1,2)
print(obj_pts)
pts = []
print("[INFO] assign four points in format:[upleft,upright,downleft,downright]")
def draw_circle(event,x,y,flags,param): #鼠标事件回调
    global pts
    if event==cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),10,(255,0,0),-1)
        pts.append([x,y])
        print('[INFO]You have assigned:',len(pts),'/ 4 points.')
img = cv2.imread(img_path)  #此处改为自己文件位置
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    if len(pts) == 4:
        break
    if cv2.waitKey(20)&0xFF=='q':  #按ESC退出
        break
cv2.destroyAllWindows()
print('Points got:\n',pts,'\n')
w = img.shape[0]
h = img.shape[1]
pts = np.array(pts,dtype ='float32').reshape(-1,1,2)
point1 = pts
point2 = np.array([[28, 0], [450, 2], [30, 371], [452, 374]],dtype = "float32")
M = cv2.getPerspectiveTransform(point1,point2)
out_pts = cv2.perspectiveTransform(obj_pts, M)
print('transformed points:\n',out_pts,'\n\n')
mapimg=cv2.imread(map_path)
for i in range(len(out_pts)):
	point=np.squeeze(out_pts[i],0)
	point = tuple(point)
	cv2.circle(mapimg,point,10,(0,0,255),-1)
cv2.imshow('map',mapimg)
cv2.waitKey(0)











