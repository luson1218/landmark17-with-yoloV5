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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def YoloV5_predete(img,conf_thres,iou_thres,Yolo5face,Yolo5landmarks,model,R_Yolov5_len):
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
	            #if x1 <= nx <= x2 and y1 <= ny <= y2:
	            #    R_Yolov5_get=R_Yolov5_get+1
	            Yolo5face.append(xyxy)
	            Yolo5landmarks.append(landmarks)
	R_Yolov5_len=R_Yolov5_len+len(Yolo5face)
	return Yolo5face,Yolo5landmarks,R_Yolov5_len             

def average_euclidean_distance(matrix1, matrix2):
    # 計算兩組矩陣之間的歐式距離
    distances = np.linalg.norm(matrix1 - matrix2, axis=1)
    
    # 計算平均距離
    average_distance = np.mean(distances)
    
    return average_distance

def FPLD_predete(img,transform,pfld_backbone):
        height, width = img.shape[:2]
        bounding_boxes, landmarks = detect_faces(img)
        Fpld_rectangle=[]
        matrix4=[]
        for box in bounding_boxes:
            x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
            #print(x1, y1, x2, y2,cx,cy)
            Fpld_rectangle=np.array([[x1,y1,x2,y2]])
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
            for (x, y) in pre_landmark.astype(np.int32):
                #cv2.circle(img, (x1 + x, y1 + y), 1, (0, 255, 255))
                points4=[x1 + x, y1 + y]
                points4s.append(points4)
            matrix4=np.array(points4s, dtype=object)
        return Fpld_rectangle,matrix4
                

def SetPoint(image,detector_dlib,predictor_dlib,face_cascade,best_network,R_dlib_len,R_CNN_len):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image = image
    height, width,_ = image.shape
    face_rects, scores, idx = detector_dlib.run(grayscale_image, 0)
    faces = face_cascade.detectMultiScale(grayscale_image, 1.1, 4)
    						#[[224, 70, 295, 295]]原始
    #print(faces)				#[[68 210 300 300]]
    #print(face_rects)				#rectangles[[(61, 233) (370, 542)]]
    #result = [[d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top()] for d in face_rects]
    #print(result)
    #faces=np.array(result)
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
        matrix3=np.array(landmarks)
        #print("matrix3{}",matrix3)
        for (x, y) in landmarks.astype(np.int32):
            cv2.circle(display_image, (x, y), 2, (0, 0, 255), 2)								# CNN
        #print(R_CNN_get,len(faces))        
        
        '''
        for i in landmarks:
            # 取得座標點的 x 和 y 值
            x, y = int(landmarks[i][0]),int(landmarks[i][1])
            #x = int(landmarks[i:])
            #y = int(landmarks[i:])
            
            # 使用 cv2.circle 繪製圓圈
            #cv2.circle(display_image, (x, y), radius, color, thickness)
            cv2.circle(display_image, (x, y), 2, ( 255, 0, 0), 1)
        '''    
    
        #for landmarks in all_landmarks:
         #   plt.scatter(landmarks[:,0], landmarks[:,1], c = 'y', s = 3)
          #  cv2.circle(display_image,(shape.part(i).x,shape.part(i).y), 3,( 0, 255, 0), 3)

    R_CNN_len=R_CNN_len+len(faces)
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        #text = " %2.2f ( %d )" % (scores[i], idx[i])
        #print(text)
        #繪製出偵測人臉的矩形範圍
        cv2.rectangle(display_image, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2. LINE_AA)
        #給68特徵點辨識取得一個轉換顏色的frame
        landmarks_frame = cv2.cvtColor(display_image, cv2. COLOR_BGR2RGB)
        #landmarks_frame = cv2.cvtColor(show_img, cv2. COLOR_GRAY2BGR)
        #找出特徵點位置
        shape = predictor_dlib(landmarks_frame, d)
        #print(shape.part.x,shape.part.y)
        
        #繪製68個特徵點
        for i in range( 68):
            cv2.circle(display_image,(shape.part(i).x,shape.part(i).y), 2,( 0, 255, 0), 2)			#Dlib
        
            # 提取特徵點矩陣
            points2 = np.array([[p.x, p.y] for p in shape.parts()])
        #print(R_dlib_get,len(face_rects))
        #matrix2 = [(int(x), int(y)) for x, y in points2]

        matrix2=np.array(points2)
        #print("matrix2{}",matrix2)
        
    
    R_dlib_len=R_dlib_len+len(face_rects)
    return display_image,matrix2,matrix3,R_dlib_len,R_CNN_len

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
	R_dlib_len=0
	R_Yolov5_get=0
	R_Yolov5_len=0
	R_CNN_len=0
	dist1_all=0
	dist2_all=0
	R_runTimes=0
	R_all=0
	fps = 0.0
	tic = time.time()
	timeo=tic
	# 設置輸出AVI文件的規格
	width, height = 640, 480
	fps = 30
	
	
	model = load_model("weights/mark17_18.pt", device)

	# 創建VideoWriter對象
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	out = cv2.VideoWriter("output2.avi", fourcc, fps, (width, height))
	text = "A:%d, Y:%d, G:%d, R:%d FPS: %2.2f"%(R_runTimes,R_FPLD_get,R_dlib_get, R_CNN_get,fps)
	#######################################################################
	weights_path = './weights/face_landmarks34_230715.pth'
	frontal_face_cascade_path = 'weights/haarcascade_frontalface_default.xml'
	#######################################################################
	#取得預設的臉部偵測器
	detector_dlib = dlib.get_frontal_face_detector()
	#根據shape_predictor方法載入68個特徵點模型，此方法為人臉表情識別的偵測器
	predictor_dlib = dlib.shape_predictor( 'weights/shape_predictor_68_face_landmarks.dat')
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

	#cap = cv2.VideoCapture(0)
	cap = cv2.VideoCapture("output.avi")
	#image = cv2.imread(image_path)
	#landmarks = best_network
	paused = False
	while(cap.isOpened()):
		timen = time.time()  # 開始時間
		ret,image1 = cap.read()
		image = cv2.flip(image1, 1)		#水平翻轉
		
		Yolo5face=[]
		Yolo5landmarks=[]

		Yolo5face,Yolo5landmarks,R_Yolov5_len=YoloV5_predete(image,0.5,0.5,Yolo5face,Yolo5landmarks,model,R_Yolov5_len)
				
		
		matrix4=[]
		Fpld_rectangle,matrix4=FPLD_predete(image,transform,pfld_backbone)
		#print(Fpld_rectangle,matrix4)
		image,matrix2,matrix3,R_dlib_len,R_CNN_len=SetPoint(image,detector_dlib,predictor_dlib,face_cascade,best_network,R_dlib_len,R_CNN_len)
				
		#for a,b,c,d in Fpld_rectangle:
		 #   cv2.rectangle(image, (a, b), (c, d), ( 0, 255, 255), 4, cv2. LINE_AA)
		R_runTimes=	R_runTimes+1	
		#if matrix4!=[]:
		Yolo5face=np.array(Yolo5face, dtype=np.int32)
		if Yolo5face!=[]:
			R_Yolov5_get=R_Yolov5_get+1
			for a,b,c,d in Yolo5face:
				cv2.rectangle(image, (a, b), (c, d), ( 255, 255, 255), 4, cv2. LINE_AA)				#yolov5
				
		Yolo5landmarks=np.array(Yolo5landmarks, dtype=np.int32)
		Yolo5landmarks = Yolo5landmarks.reshape(-1, 2)
		if Yolo5landmarks!=[]:
			for (x, y) in Yolo5landmarks.astype(np.int32):
				cv2.circle(image, (x, y), 2, (255, 255, 255), 2)									#yolov5
		
		if not np.array_equal(matrix4, []):
		    R_FPLD_get=R_FPLD_get+1
		    for (x, y) in matrix4.astype(np.int32):
		        cv2.circle(image, (x, y), 2, (0, 255, 255), 2)									#FPLD
				
		if not np.array_equal(matrix2, []):
		    R_dlib_get=R_dlib_get+1
		    for (x, y) in matrix2.astype(np.int32):
		        cv2.circle(image, (x, y), 2, (0, 255, 0), 2)									#Dlib
				
		if not np.array_equal(matrix3, []):
		    R_CNN_get=R_CNN_get+1
		    for (x, y) in matrix3.astype(np.int32):
		        cv2.circle(image, (x, y), 2, (0, 0, 255), 2)									#CNN
		
		text = "Yellow(PFLD):%d,Green(Dlib):%d,Red(CNN):%d,White(YoloV5_mark17):%d"%(R_FPLD_get,R_dlib_get, R_CNN_get,R_Yolov5_get)
		cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,(32, 32, 32), 4, cv2.LINE_AA)		#黑底
		cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,(240, 240, 240), 1, cv2.LINE_AA)		#白字
		
		text = "Total:%d, FPS:%2.2f"%(R_runTimes,fps)
		cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.0,(32, 32, 32), 4, cv2.LINE_AA)		#黑底
		cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.0,(240, 240, 240), 1, cv2.LINE_AA)		#白字
		
		
		toc = time.time()
		curr_fps = 1.0 / (toc - tic)
		# calculate an exponentially decaying average of fps number
		#print(toc - tic)
		fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
		tic = toc
		'''
		if R_runTimes>fps:
			text = "A:%d, Y:%d, G:%d, R:%d FPS: %2.2f"%(R_runTimes,R_FPLD_get,R_dlib_get, R_CNN_get,fps)
			cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,(32, 32, 32), 4, cv2.LINE_AA)		#黑底
			cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,(240, 240, 240), 1, cv2.LINE_AA)		#白字
		
			R_runTimes=0
			R_FPLD_get=0
			R_dlib_get=0
			R_CNN_get=0
        '''        
		'''
		cv2.namedWindow("Frame", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
		cv2.resizeWindow("Frame", 1024, 968)    # 设置长和宽
		'''
		out.write(image)
		cv2.imshow("Frame", image)
		key =cv2.waitKey(1) 			#1mes
		if key == ord(' '):  # 空格鍵
			paused = not paused
			if paused:
				#print("Video paused")
				key =cv2.waitKey(0) 
				paused=False
		elif key == ord('q'):  # 空格鍵
		    break		
		elif key == 27:	#'ESC'
		    break		
	cap.release()
	cv2.destroyAllWindows()
		
	
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
FPLD(Y):744    All : 1008					checkpoint.pth.tar
Dlib(G):949,   197.772600724652				shape_predictor_68_face_landmarks.dat
CNN(R):953,   225.71406060376262			face_landmarks34_230715.pth

lfpw/trainset/image_0854_mirror.jpg
FPLD(Y):744    All : 1008										checkpoint_epoch_459.pth.tar
Dlib(G):949/1361,   2.6876294616271257/7.50088035697309			shape_predictor_68_face_landmarks.dat
CNN(R):953/2471,   1.8892849326245704/12.305378789094538		face_landmarks34_230715.pth

lfpw/trainset/image_0854_mirror.jpg
FPLD(Y):744    All : 1008										checkpoint_epoch_459.pth.tar
Dlib(G):949/1361,   2.3650832497531473/10.112219430536149		all_predictor.dat
CNN(R):953/2471,   3.081694139528725/17.013933249836903			face_landmarks34_230516.pth

'''

    
    
    

