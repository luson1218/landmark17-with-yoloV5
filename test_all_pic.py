# USAGE
# python evaluate_shape_predictor.py --predictor predictor.dat --xml ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml
# python test_show_pic.py --xml ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml
# python test_show_pic.py -x ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml
# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import cv2

import argparse
import dlib
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import re

from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
from Fclass import Transforms,FaceLandmarksDataset,Network

import torchvision
from models.pfld import PFLDInference, AuxiliaryNet
from mtcnn2.detector import detect_faces
from deepface import DeepFace

from pathlib import Path
import sys
import os
import copy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg後端（需安裝tkinter庫）
import matplotlib.pyplot as plt

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    #coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    #coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    #coords[:, :10] /= gain
    coords[:, [0, 2, 4, 6, 8,10, 12, 14, 16, 18,20, 22, 24, 26, 28,30, 32]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9,11, 13, 15, 17, 19,21, 23, 25, 27, 29,31, 33]] -= pad[1]  # y padding
    coords[:, :34] /= gain

    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5

    coords[:, 10].clamp_(0, img0_shape[1])  # x1
    coords[:, 11].clamp_(0, img0_shape[0])  # y1
    coords[:, 12].clamp_(0, img0_shape[1])  # x2
    coords[:, 13].clamp_(0, img0_shape[0])  # y2
    coords[:, 14].clamp_(0, img0_shape[1])  # x3
    coords[:, 15].clamp_(0, img0_shape[0])  # y3
    coords[:, 16].clamp_(0, img0_shape[1])  # x4
    coords[:, 17].clamp_(0, img0_shape[0])  # y4
    coords[:, 18].clamp_(0, img0_shape[1])  # x5
    coords[:, 19].clamp_(0, img0_shape[0])  # y5

    coords[:, 20].clamp_(0, img0_shape[1])  # x1
    coords[:, 21].clamp_(0, img0_shape[0])  # y1
    coords[:, 22].clamp_(0, img0_shape[1])  # x2
    coords[:, 23].clamp_(0, img0_shape[0])  # y2
    coords[:, 24].clamp_(0, img0_shape[1])  # x3
    coords[:, 25].clamp_(0, img0_shape[0])  # y3
    coords[:, 26].clamp_(0, img0_shape[1])  # x4
    coords[:, 27].clamp_(0, img0_shape[0])  # y4
    coords[:, 28].clamp_(0, img0_shape[1])  # x5
    coords[:, 29].clamp_(0, img0_shape[0])  # y5

    coords[:, 30].clamp_(0, img0_shape[1])  # x1
    coords[:, 31].clamp_(0, img0_shape[0])  # y1
    coords[:, 32].clamp_(0, img0_shape[1])  # x2
    coords[:, 33].clamp_(0, img0_shape[0])  # y2
    return coords
#================================================================
# Convert
def YoloV5_predete(img,conf_thres,iou_thres,Yolo5face,Yolo5landmarks,model,nx,ny,R_Yolov5_get,R_Yolov5_len):
	img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
	img = np.ascontiguousarray(img)				#將 img 轉換為連續內存的 numpy 數組。
	if len(img.shape) == 4:
	    orgimg = np.squeeze(img.transpose(0, 2, 3, 1), axis= 0)
	else:
	    orgimg = img.transpose(1, 2, 0)
	#print(orgimg.shape)         
	orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
	img0 = copy.deepcopy(orgimg)
	h0, w0 = orgimg.shape[:2]  # orig hw
	img_size=max(h0, w0)
	imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size  強至補成32倍數
	img = letterbox(img0, new_shape=imgsz)[0]
	img = img.transpose(2, 0, 1).copy()			#這是一個將圖像的通道順序由寬、高、通道（w,h,c）轉換為通道、寬、高（c,w,h）的操作。
	img = torch.from_numpy(img).to(device)
	img = img.float()  # uint8 to fp16/32
	img /= 255.0  # 0 - 255 to 0.0 - 1.0
	if img.ndimension() == 3:	#檢查 img 的維度是否為 3（也就是是否是一張單獨的圖像，而不是一個 batch）
	    img = img.unsqueeze(0)	#將其轉換成一個包含單個元素的 Tensor（也就是將維度擴展成 [1, c, w, h] 的格式），這樣就形成了一個包含單張圖像的 batch。
	    #print('--3--',img.shape)
	        
	# Inference
	pred = model(img)[0]
	#print(pred)
	pred = non_max_suppression_face(pred, conf_thres, iou_thres)
	#print('--4--',len(pred[0]))
	for i, det in enumerate(pred):  # detections per image
	            
	    if len(det):
	        # Rescale boxes from img_size to im0 size
	        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
	
	        # Print results
	        for c in det[:, -1].unique():
	            n = (det[:, -1] == c).sum()  # detections per class
	
	        det[:, 5:39] = scale_coords_landmarks(img.shape[2:], det[:, 5:39], img0.shape).round()

	        for j in range(det.size()[0]):
	            xyxy = det[j, :4].view(-1).tolist()
	            conf = det[j, 4].cpu().numpy()
	            landmarks = det[j, 5:39].view(-1).tolist()
	            class_num = det[j, 39].cpu().numpy()
	            #print('--5--',xyxy,conf,landmarks,class_num)
	            x1, y1, x2, y2 = xyxy
	            if x1 <= nx <= x2 and y1 <= ny <= y2:
	                R_Yolov5_get=R_Yolov5_get+1
	            Yolo5face.append(xyxy)
	            Yolo5landmarks.append(landmarks)
	R_Yolov5_len=R_Yolov5_len+len(Yolo5face)
	return Yolo5face,Yolo5landmarks,R_Yolov5_get,R_Yolov5_len             

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
def average_euclidean_distance(matrix1, matrix2):
    # 計算兩組矩陣之間的歐式距離
    distances = np.linalg.norm(matrix1 - matrix2, axis=1)
    
    # 計算平均距離
    average_distance = np.mean(distances)
    
    return average_distance
    
    

def FPLD_predete(img,transform,pfld_backbone,nx,ny,R_FPLD_get,R_FPLD_len):
        Fpld_rectangles=[]
        height, width = img.shape[:2]
        bounding_boxes, landmarks = detect_faces(img)
        #bounding_boxes = np.round(bounding_boxes_u).astype(np.int)
        #print(len(bounding_boxes),bounding_boxes)
        Fpld_rectangle=[]
        matrix4=[]
        for box in bounding_boxes:
            x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
            #print(x1, y1, x2, y2,cx,cy)
            
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            cx = x1 + w // 2
            cy = y1 + h // 2

            size = int(max([w, h]) * 1.1)
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            edx1 = max(0, -x1)
            edy1 = max(0, -y1)
            edx2 = max(0, x2 - width)
            edy2 = max(0, y2 - height)
            
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                R_FPLD_get=R_FPLD_get+1
                #print(R_FPLD_get)
            

            cropped = img[y1:y2, x1:x2]
            if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
                cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2,
                                             cv2.BORDER_CONSTANT, 0)

            input = cv2.resize(cropped, (112, 112))
            input = transform(input).unsqueeze(0).to(device)
            _, landmarks = pfld_backbone(input)
            pre_landmark = landmarks[0]
            pre_landmark = pre_landmark.cpu().detach().numpy().reshape(
                -1, 2) * [size, size] - [edx1, edy1]
            points4s=[]
            points4=[]
            #繪製出偵測人臉的矩形範圍
            #cv2.rectangle(img, (x1, y1), (x2, y2), ( 0, 255, 255), 4, cv2. LINE_AA)
            
            for (x, y) in pre_landmark.astype(np.int32):
                #cv2.circle(img, (x1 + x, y1 + y), 2, (0, 255, 255))
                points4=[x1 + x, y1 + y]
                points4s.append(points4)
            matrix4=np.array(points4s, dtype=object)
            Fpld_rectangle=[x1,y1,x2,y2]
            Fpld_rectangles.append(Fpld_rectangle)
            
        Fpld_rectMax=np.array(Fpld_rectangles, dtype=object)
        R_FPLD_len=R_FPLD_len+len(bounding_boxes)
        return Fpld_rectMax,matrix4,R_FPLD_get,R_FPLD_len
                

def SetPoint(image,detector_dlib,predictor_dlib,face_cascade,best_network,cx,cy,R_dlib_get,R_CNN_get,R_dlib_len,R_CNN_len):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image = image
    height, width,_ = image.shape
    face_rects, scores, idx = detector_dlib.run(grayscale_image, 0)
    time4 = time.time()
    faces = face_cascade.detectMultiScale(grayscale_image, 1.1, 4)
    time5 = time.time()
    matrix2=[]
    matrix3=[]
    all_landmarks = []
    for (x, y, w, h) in faces:
        image = grayscale_image[y:y+h, x:x+w]
        image = TF.resize(Image.fromarray(image), size=(224, 224))
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])

        with torch.no_grad():
            landmarks = best_network(image.unsqueeze(0)) 

        landmarks = (landmarks.view(68,2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
        all_landmarks.append(landmarks)
        #print(landmarks)
        cv2.rectangle(display_image, (x, y), (x+w, y+h), ( 0, 0, 255), 4, cv2. LINE_AA)
        bgetone=0
        if x <= cx <= x+w and y <= cy <= y+h:
                R_CNN_get=R_CNN_get+1
                bgetone=1
        if bgetone:
            matrix3=np.array(landmarks)
            #print(matrix3)
        
        for (x, y) in landmarks.astype(np.int32):
            cv2.circle(display_image, (x, y), 2, (0, 0, 255), 2)								# CNN

    R_CNN_len=R_CNN_len+len(faces)
    
    
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        text = " %2.2f ( %d )" % (scores[i], idx[i])
        #print(text)
        #繪製出偵測人臉的矩形範圍
        cv2.rectangle(display_image, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2. LINE_AA)
        bgetone=0
        if x1 <= cx <= x2 and y1 <= cy <= y2:
                R_dlib_get=R_dlib_get+1
                bgetone=1
        #給68特徵點辨識取得一個轉換顏色的frame
        landmarks_frame = cv2.cvtColor(display_image, cv2. COLOR_BGR2RGB)
        #landmarks_frame = cv2.cvtColor(show_img, cv2. COLOR_GRAY2BGR)
        #找出特徵點位置
        shape = predictor_dlib(landmarks_frame, d)
        #print(shape.part.x,shape.part.y)
        
        #繪製68個特徵點
        points2=[]
        for i in range( 68):
            cv2.circle(display_image,(shape.part(i).x,shape.part(i).y), 2,( 0, 255, 0), 2)			#Dlib
        
            # 提取特徵點矩陣
            points2 = np.array([[p.x, p.y] for p in shape.parts()])
        #print(R_dlib_get,len(face_rects))
        matrix2=[]
        if bgetone:
            matrix2=np.array(points2)
            #print(matrix2)
    
    
    
    R_dlib_len=R_dlib_len+len(face_rects)
    return display_image,matrix2,matrix3,R_dlib_get,R_CNN_get,R_dlib_len,R_CNN_len,time4,time5

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model



def main(args):
	# construct the argument parser and parse the arguments
	#ap = argparse.ArgumentParser()
	#ap.add_argument("-x", "--xml", required=True,
	#help="path to input training/testing XML file")
	#args = vars(ap.parse_args())

	#print(args)
	allpic=0
	average_dist_all=0
	R_FPLD_get=0
	R_dlib_get=0
	R_CNN_get=0
	R_Yolov5_get=0
	R_Yolov5_len=0
	R_dlib_len=0
	R_CNN_len=0
	R_FPLD_len=0
	dist1_all=0
	dist2_all=0
	R_all=0
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#model = load_model("weights/yolov5m-face.pt", device)
	model = load_model("weights/mark17_18.pt", device)


	#######################################################################
	#image_path = 'pic1.jpg'
	weights_path = './weights/face_landmarks34_230715.pth'
	frontal_face_cascade_path = 'weights/haarcascade_frontalface_default.xml'
	#######################################################################
	#取得預設的臉部偵測器
	detector_dlib = dlib.get_frontal_face_detector()
	#根據shape_predictor方法載入68個特徵點模型，此方法為人臉表情識別的偵測器
	predictor_dlib = dlib.shape_predictor( 'weights/shape_predictor_68_face_landmarks.dat')
	#predictor_dlib = dlib.shape_predictor( './weights/shape_predictor_5_face_landmarks.dat')
	#predictor_dlib = dlib.shape_predictor( 'all_predictor.dat')

	face_cascade = cv2.CascadeClassifier(frontal_face_cascade_path)

	best_network = Network()
	best_network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))) 
	best_network.eval()

	checkpoint = torch.load(args.model_path, map_location=device)
	pfld_backbone = PFLDInference().to(device)
	pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
	pfld_backbone.eval()
	pfld_backbone = pfld_backbone.to(device)
	transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])    

	#LANDMARKS = set(list(range(36, 48)))
	# to easily parse out the eye locations from the XML file we can
	# utilize regular expressions to determine if there is a 'part'
	# element on any given line
	PART = re.compile("part name='[0-9]+'")
	
	time_yolo_all=0
	time_FPLD_all=0
	time_dlib_all=0
	time_cnn_all =0

	#rows = open(args["xml"]).read().strip().split("\n")
	rows = open(args.xml).read().strip().split("\n")
	#print(rows)    
	points = []
	image_name=''
	# loop over the rows of the data split file
	for row in rows:
		# check to see if the current line has the (x, y)-coordinates for
		# the facial landmarks we are interested in
		key = cv2.waitKey(1) & 0xFF
		if key == ord("e"):
			break
		parts = re.findall(PART, row)
		#print(parts)
		# if there is no information related to the (x, y)-coordinates of
		# the facial landmarks, we can write the current line out to disk
		# with no further modifications
		
		if len(parts) == 0:
			
			#print(points_array) 
			#points = []
			attr = "image file='"
			i = row.find(attr)
			j = row.find("'", i + len(attr))
			#print(i,j)
			name = (row[i + len(attr):j])
			
			if i>0:
				image_name=name
				#print(image_name)
			
			#<box top='224' left='70' width='295' height='295'>
			attr = "top='"
			x = row.find(attr)
			if x>0:
			    boxs=[]
			    i = row.find(attr)
			    j = row.find("'", i + len(attr) + 1)
			    top = int(row[i + len(attr):j])
			    attr = "left='"
			    i = row.find(attr)
			    j = row.find("'", i + len(attr) + 1)
			    left = int(row[i + len(attr):j])
			    attr = "width='"
			    i = row.find(attr)
			    j = row.find("'", i + len(attr) + 1)
			    width = int(row[i + len(attr):j])
			    attr = "height='"
			    i = row.find(attr)
			    j = row.find("'", i + len(attr) + 1)
			    height = int(row[i + len(attr):j])
			    boxs.append([left, top, left+width, top+height])
			    #print(boxs)
		
		
		
			
			
#===========================================================================================================

		
		# otherwise, there is annotation information that we must process
		else:
			# parse out the name of the attribute from the row
		
			attr = "name='"
			i = row.find(attr)
			j = row.find("'", i + len(attr) + 1)
			name = int(row[i + len(attr):j])
			attr = "x='"
			i = row.find(attr)
			j = row.find("'", i + len(attr) + 1)
			x = int(row[i + len(attr):j])
			attr = "y='"
			i = row.find(attr)
			j = row.find("'", i + len(attr) + 1)
			y = int(row[i + len(attr):j])
			#x = float(row.find("x="))
	    		#y = float(row.find("y="))
			points.append([x, y])
			#print(row)
			#print(name)
			#print(x)
			#print(y)
			#if name==67 and number==0:
			if name==67:
				points_array = np.array(points)
				R_all=R_all+1
				# 讀取圖像
				#image = dlib.load_rgb_image("ibug_300W_large_face_landmark_dataset/"+name)
				print(image_name)
				img_path=("ibug_300W_large_face_landmark_dataset/"+image_name)
				#image = cv2.imread("ibug_300W_large_face_landmark_dataset/"+image_name)
				#image = cv2.imread(img_path)
				orgimg = cv2.imread(img_path)  # BGR
				assert orgimg is not None, 'Image Not Found ' + path
				# Padded resize
				h0, w0 = orgimg.shape[:2]  # orig hw
				img_size = 640																###############32 X
				image= letterbox(orgimg, new_shape=img_size)[0]
				h1, w1 = image.shape[:2]  # orig hw
				scale_factor = img_size / max(h0, w0)  # resize image to img_size
				#print(h0,w0,h1,w1)
				#print(scale_factor)
				
				    
				
				
				#boxs = boxs.float()
				boxs_float = [[float(x) for x in box] for box in boxs]
				#boxs = boxs.astype(np.float)
				boxs_float_scaled = [[x * scale_factor for x in box] for box in boxs_float]
				#boxs = boxs.astype(np.int32)
				
				points_array_float=[[float(x) for x in points_array_] for points_array_ in points_array]
				points_array_float_scaled = [[x * scale_factor for x in points_array_] for points_array_ in points_array_float]
				points_array=[[int(x) for x in points_array_] for points_array_ in points_array_float_scaled]
				
				#print(boxs)
				#print(boxs_float)
				#print(boxs_float_scaled)
				#print(boxs_float_scaled[0][0])
				#print(boxs_float_scaled[0][1])
				#print(boxs_float_scaled[0][2])
				#print(boxs_float_scaled[0][3])
				#h=(abs(w0-h0)/2)*scale_factor
				#x=(abs(h0-w0)/2)*scale_factor
				add_y=int((h1-h0*scale_factor)/2)
				add_x=int((w1-w0*scale_factor)/2)
				#print('x=',add_x,'y=',add_y)
				
				
				if h0<w0:			#補高
				    #h=(h1-h0*scale_factor)/2
				    #h=((w0-h0)/2)*scale_factor
				    #if scale_factor>1:
				    boxs_float_scaled[0][1]=boxs_float_scaled[0][1]+add_y			#增加
				    boxs_float_scaled[0][3]=boxs_float_scaled[0][3]+add_y			#增加
				    #print('h=',h)
				    #else:
				     #   boxs_float_scaled[0][1]=boxs_float_scaled[0][1]-h			#減少
				      #  boxs_float_scaled[0][3]=boxs_float_scaled[0][3]-h			#減少
				    				
				else:				#補寬 
				    #x=(w1-w0*scale_factor)/2
				    #x=((h0-w0)/2)*scale_factor
				    #if scale_factor>1:
				    boxs_float_scaled[0][0]=boxs_float_scaled[0][0]+add_x			#增加
				    boxs_float_scaled[0][2]=boxs_float_scaled[0][2]+add_x			#增加
				    #print('x=',x)
				    #else:
				     #   boxs_float_scaled[0][0]=boxs_float_scaled[0][0]-x			#減少
				      #  boxs_float_scaled[0][2]=boxs_float_scaled[0][2]-x			#減少
				
				#print(boxs_float_scaled)
				boxs=[[int(x) for x in boxs_float_scaledi] for boxs_float_scaledi in boxs_float_scaled]
				#print(boxs)
				
				for a,b,c,d in boxs:
				    cx=(a+c)/2
				    cy=(b+d)/2
				    #print(a,b,c,d)
				
				
				
				
				
				    #print(boxs,cx,cy)
				
				
				Yolo5face=[]
				Yolo5landmarks=[]
				time1 = time.time()
				Yolo5face,Yolo5landmarks,R_Yolov5_get,R_Yolov5_len=YoloV5_predete(image,0.5,0.5,Yolo5face,Yolo5landmarks,model,cx,cy,R_Yolov5_get,R_Yolov5_len)
				#print('--2--',Yolo5face)
				#print('--3--',Yolo5landmarks)
				time2 = time.time()

				matrix4=[]
				#faces0 = DeepFace.extract_faces(img_path,detector_backend= backends[2],enforce_detection=False)	#'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface'
				Fpld_rectangle,matrix4,R_FPLD_get,R_FPLD_len=FPLD_predete(image,transform,pfld_backbone,cx,cy,R_FPLD_get,R_FPLD_len)
				#print(Fpld_rectangle,matrix4)
				time3 = time.time()
				image,matrix2,matrix3,R_dlib_get,R_CNN_get,R_dlib_len,R_CNN_len,time4,time5=SetPoint(image,detector_dlib,predictor_dlib,face_cascade,best_network,cx,cy,R_dlib_get,R_CNN_get,R_dlib_len,R_CNN_len)
				
				time_yolo=time2-time1
				time_FPLD=time3-time2
				time_dlib=time4-time3
				time_cnn =time5-time4
				
				time_yolo_all=time_yolo_all+time_yolo
				time_FPLD_all=time_FPLD_all+time_FPLD
				time_dlib_all=time_dlib_all+time_dlib
				time_cnn_all =time_cnn_all+time_cnn
				
				
				#'''
				for a,b,c,d in Fpld_rectangle:
				    cv2.rectangle(image, (a, b), (c, d), ( 0, 255, 255), 6, cv2. LINE_AA)
				
				Yolo5face=np.array(Yolo5face, dtype=np.int32)
				for a,b,c,d in Yolo5face:
				    cv2.rectangle(image, (a, b), (c, d), ( 255, 255, 255), 4, cv2. LINE_AA)				#yolov5
				
				
				if matrix4!=[]:
				    for (x, y) in matrix4.astype(np.int32):
				        cv2.circle(image, (x, y), 2, (0, 255, 255), 2)									#FPLD
				
				Yolo5landmarks=np.array(Yolo5landmarks, dtype=np.int32)
				Yolo5landmarks = Yolo5landmarks.reshape(-1, 2)
				
				if Yolo5landmarks!=[]:
				    for (x, y) in Yolo5landmarks.astype(np.int32):
				        cv2.circle(image, (x, y), 2, (255, 255, 255), 2)									#yolov5
				
				
				
				points=[]
				for a,b,c,d in boxs:
				    cv2.rectangle(image, (a, b), (c, d), ( 255, 0, 0), 4, cv2. LINE_AA)
			    
				for (x, y) in points_array:
				    cv2.circle(image, (x+add_x, y+add_y), 2, (255, 0, 0), 2)							#Source
				
				#'''
				
				
				
				
				

			
				points=[]
				matrix1=np.array(points_array)
				# 使用函數計算平均歐式距離
				print("FPLD(Y)  :{}/{},Time{}/{},    All : {}".format(R_FPLD_get,R_FPLD_len,time_FPLD,time_FPLD_all,R_all))
				print("Yolov5(W):{}/{},Time{}/{}".format(R_Yolov5_get,R_Yolov5_len,time_yolo,time_yolo_all))
				if matrix2!=[]:
				    average_dist1 = average_euclidean_distance(matrix1, matrix2)
				    dist1_all=dist1_all+average_dist1
				    average_dist1_all=dist1_all/R_dlib_get
				    
				    
				if matrix3!=[]:
				    average_dist2 = average_euclidean_distance(matrix1, matrix3)
				    dist2_all=dist2_all+average_dist2
				    average_dist2_all=dist2_all/R_CNN_get
				    
				
				print("Dlib(G)  :{}/{},Time{}/{},   {}/{}".format(R_dlib_get,R_dlib_len,time_dlib,time_dlib_all,average_dist1,average_dist1_all))
				print("CNN(R)   :{}/{},Time{}/{},   {}/{}".format(R_CNN_get,R_CNN_len,time_cnn,time_cnn_all,average_dist2,average_dist2_all))
				#'''
				cv2.namedWindow("Frame", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
				cv2.resizeWindow("Frame", 1024, 968)    # 设置长和宽
				cv2.imshow("Frame", image)
				if cv2.waitKey(0) == 27:	#'ESC'
				    break
				#'''

	error = dlib.test_shape_predictor(args["xml"], args["predictor"])
	print("[INFO] error: {}".format(error))
	
def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--xml',
                        default="./ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml",
                        type=str)
    parser.add_argument('--model_path',
                        default="./weights/checkpoint.pth.tar",
                        #default="./checkpoint_epoch_459.pth.tar",
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
'''    
lfpw/trainset/image_0854_mirror.jpg
[[267, 65, 372, 169]] 319.5 117.0
FPLD(Y):811/1504,    All : 1008					checkpoint.pth.tar
Dlib(G):949,   197.772600724652				shape_predictor_68_face_landmarks.dat
CNN(R):953,   225.71406060376262			face_landmarks34_230715.pth

lfpw/trainset/image_0854_mirror.jpg
FPLD(Y):811/1504,    All : 1008	                                    allpic:2500,Average Euclidean Distance: 3.54,  average:3.33
Dlib(G):949/1361,   2.6876294616271257/7.50088035697309			shape_predictor_68_face_landmarks.dat
CNN(R):953/2471,   1.8892849326245704/12.305378789094538		face_landmarks34_230715.pth

lfpw/trainset/image_0854_mirror.jpg
FPLD(Y):811/1504,    All : 1008										checkpoint_epoch_459.pth.tar
Dlib(G):949/1361,   2.3650832497531473/10.112219430536149		all_predictor.dat
CNN(R):953/2471,   3.081694139528725/17.013933249836903			face_landmarks34_230516.pth





lfpw/trainset/image_0854_mirror.jpg
FPLD(Y)  :811/1504,    All : 1008
Yolov5(W):769/1499												yolov5n-face.pt
Dlib(G)  :948/1359,   4.003764653799201/6.649432671914636
CNN(R)   :953/2471,   1.8892849326245704/12.936084564578968



FPLD(Y)  :847/1344,Time0.0352933406829834/44.12041711807251,    All : 1008					640
Yolov5(W):1006/1830,Time0.011935710906982422/13.463255643844604									yolov5s-face.pt
Dlib(G)  :927/1103,Time0.025972604751586914/29.201749801635742,   5.808435291741833/12.709603802992644
CNN(R)   :945/1631,Time0.20797419548034668/161.40063977241516,   19.624630832751592/15.863672257278955

FPLD(Y)  :847/1344,Time0.03621172904968262/44.239763259887695,    All : 1008				640
Yolov5(W):1008/1747,Time0.015363454818725586/15.633798122406006									yolov5n-face.pt
Dlib(G)  :927/1103,Time0.02589106559753418/28.910481452941895,   5.808435291741833/12.709603802992644
CNN(R)   :945/1631,Time0.20774126052856445/160.63973331451416,   19.624630832751592/15.863672257278955

FPLD(Y)  :847/1344,Time0.03692936897277832/43.89257001876831,    All : 1008					640
Yolov5(W):1006/1846,Time0.01883697509765625/19.398738145828247									yolov5m-face.pt
Dlib(G)  :927/1103,Time0.028055429458618164/29.16134548187256,   5.808435291741833/12.709603802992644
CNN(R)   :945/1631,Time0.2182762622833252/161.35220336914062,   19.624630832751592/15.863672257278955




FPLD(Y)  :845/1401,Time0.05495429039001465/62.15014314651489,    All : 1008					960
Yolov5(W):972/1728,Time0.018266916275024414/19.259578466415405									yolov5n-face.pt
Dlib(G)  :942/1196,Time0.05618691444396973/61.07340621948242,   18.193280968482785/14.309104143579152
CNN(R)   :951/1946,Time0.40458154678344727/313.4965727329254,   29.49396896079343/18.820805119284582


FPLD(Y)  :845/1401,Time0.06398797035217285/62.308247089385986,    All : 1008				960
Yolov5(W):997/1821,Time0.036618947982788086/34.91586637496948									yolov5m-face.pt
Dlib(G)  :942/1196,Time0.06192350387573242/62.96362090110779,   18.193280968482785/14.309104143579152
CNN(R)   :951/1946,Time0.393369197845459/319.86288595199585,   29.49396896079343/18.820805119284582



FPLD(Y)  :847/1344,Time0.04029989242553711/44.757962226867676,    All : 1008
Yolov5(W):997/1409,Time0.018717527389526367/15.387505531311035									mark17_18.pt
Dlib(G)  :927/1103,Time0.029958248138427734/28.865395069122314,   5.808435291741833/12.709603802992644
CNN(R)   :945/1631,Time0.20215964317321777/160.37259435653687,   19.624630832751592/15.863672257278955



'''

    
    
    

