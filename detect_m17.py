# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import sys
import os

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy
import math

from scipy.linalg import svd


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

def cal_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance
    
def get_ro_vector(plane_points,object_pts):
	# 计算质心
	
	
	centroid_plane = np.mean(plane_points, axis=0)
	centroid_3D = np.mean(object_pts, axis=0)
	'''


	# 内部参数
	focal_length_x = 1000  # x轴焦距
	focal_length_y = 1000  # y轴焦距
	principal_point_x = 320  # 主点x坐标
	principal_point_y = 240  # 主点y坐标

	# 外部参数
	rotation_matrix = np.array([[1, 0, 0],
                            	[0, 1, 0],
                            	[0, 0, 1]])  # 3x3旋转矩阵
	translation_vector = np.array([0, 0, 0])  # 平移向量

	# 平面点坐标
	x,y=centroid_plane
	#x = 200
	#y = 150

	# Step 1: 将平面坐标转换为NDC坐标
	x_ndc = (x - principal_point_x) / focal_length_x
	y_ndc = (y - principal_point_y) / focal_length_y

	# Step 2: 使用外部参数将NDC坐标转换为立体坐标
	ndc_coordinates = np.array([x_ndc, y_ndc, 1])
	stereo_coordinates = np.dot(rotation_matrix, ndc_coordinates) + translation_vector

	X, Y, Z = stereo_coordinates

	print(f"平面坐标 ({x}, {y}) 映射到立体坐标 ({X}, {Y}, {Z})")
	'''






	delta_P=plane_points-centroid_plane			#取用＊＊
	delta_C=object_pts-centroid_3D
	#print("---------1----------")
	#print(centroid_plane)
	#print("---------2----------")
	#print(centroid_3D)
	
	#print("---------3----------")
	print(delta_P)
	#print(delta_C)
	
	'''
	
	
	# 将质心移到原点
	plane_points -= centroid_plane
	object_pts -= centroid_3D

	# 计算协方差矩阵
	print(plane_points)
	print(object_pts)
	covariance_matrix = np.dot(plane_points.T, object_pts)
	
	
	
	# 计算 P 和 C 之间的差异
	delta_P = P - centroid_P  # 其中 centroid_P 是平面点 P 的质心
	delta_C = C - centroid_C  # 其中 centroid_C 是立体点 C 的质心
	'''

	# 计算旋转向量
	rotation_vector = np.cross(delta_P, delta_C)

	# 如果需要，归一化旋转向量
	#rotation_vector /= np.linalg.norm(rotation_vector)
	
	'''


	

	# 奇异值分解
	U, S, Vt = svd(covariance_matrix)
	print(U)
	print(Vt)

	# 计算旋转矩阵
	rotation_matrix = np.dot(U, Vt)

	# 从旋转矩阵中提取旋转向量
	rotation_vector = np.zeros(3)
	rotation_vector[0] = rotation_matrix[2, 1] - rotation_matrix[1, 2]
	rotation_vector[1] = rotation_matrix[0, 2] - rotation_matrix[2, 0]
	rotation_vector[2] = rotation_matrix[1, 0] - rotation_matrix[0, 1]
	'''
	print("Rotation Vector:", rotation_vector)
	return rotation_vector


# 從旋轉向量轉換為歐拉角
def get_euler_angle(rotation_vector):
    # calculate rotation angles
    #print(rotation_vector)
    #'''
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    #print("rotation_matrix:\n",rotation_matrix)
    m21=rotation_matrix[2, 1]
    m01=rotation_matrix[0, 1]
    m11=rotation_matrix[1, 1]
    m20=rotation_matrix[2, 0]
    m22=rotation_matrix[2, 2]
    # 使用旋转矩阵来计算四元数
    w = 0.5 * np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2])
    x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * w)
    y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * w)
    z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * w)
    # 创建四元数
    quaternion = np.array([w, x, y, z])
    #print("Quaternion:", quaternion)
    #'''
    
    '''
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    # 使用旋转矩阵来计算四元数
    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2)*rotation_vector[0][0] / theta
    y = math.sin(theta / 2)*rotation_vector[1][0] / theta
    z = math.sin(theta / 2)*rotation_vector[2][0] / theta
    # 创建四元数
    quaternion = np.array([w, x, y, z])
    print("Quaternion:", quaternion)
    # roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x*x + y * y)
    roll = math.atan2(t0, t1)
    
    # pitch (y-axis rotation)
    t2 = 2.0 * (w * y - x * z)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    pitch = math.asin(t2)
    
    # yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (z*z + z * z)
    yaw = math.atan2(t3, t4)
    '''
    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x - y * z)
    if t0 > 1.0:
        t0 = 1.0
    if t0 < -1.0:
        t0 = -1.0
    pitch = math.asin(t0)
    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y + x * z)
    t5 = 1.0 - 2.0 * (y * y + ysqr)
    yaw = math.atan2(t2, t5)
    
    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)
    
    
    #pitch = math.asin(m21)
    #yaw = np.arctan2(m01, m11)
    #roll = np.arctan2(m20, m22)
    #print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
    #print("t0,t2,t5,t3,t4:",t0,t2,t5,t3,t4)
    #print("pitch,yaw,roll,math.pi:",pitch,yaw,roll,math.pi)
    
	# 單位轉換：將弧度轉換為度
    Y = int((pitch/math.pi)*180)
    X = int((yaw/math.pi)*180)
    Z = int((roll/math.pi)*180)
    
    #print('Pitch:{}, yaw:{}, roll:{}'.format(Y,X,Z))
    
    return 0, Y, X, Z

def get_euler_angle2(rotation_vector):
	# 计算旋转矩阵
	theta = np.linalg.norm(rotation_vector)
	rotation_axis = rotation_vector / theta
	cos_theta = math.cos(theta)
	sin_theta = math.sin(theta)
	x, y, z = rotation_axis

	R = np.array([
    [cos_theta + (1 - cos_theta) * x**2, (1 - cos_theta) * x * y - sin_theta * z, (1 - cos_theta) * x * z + sin_theta * y],
    [(1 - cos_theta) * y * x + sin_theta * z, cos_theta + (1 - cos_theta) * y**2, (1 - cos_theta) * y * z - sin_theta * x],
    [(1 - cos_theta) * z * x - sin_theta * y, (1 - cos_theta) * z * y + sin_theta * x, cos_theta + (1 - cos_theta) * z**2]
	])

	# 从旋转矩阵中提取欧拉角
	pitch = math.asin(-R[2][0])
	roll = math.atan2(R[1][0], R[0][0])
	yaw = math.atan2(R[2][1], R[2][2])

	# 将弧度转换为度
	pitch_degrees = math.degrees(pitch)
	roll_degrees = math.degrees(roll)
	yaw_degrees = math.degrees(yaw)

	print(f"Pitch: {pitch_degrees} degrees")
	print(f"Roll: {roll_degrees} degrees")
	print(f"Yaw: {yaw_degrees} degrees")
	return

    
def load_model(weights, device):
    #print(weights)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


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

def show_results(img, xyxy, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()
    
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255)]

    #for i in range(5):
    for i in range(17):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect(
    model,
    source,
    device,
    project,
    name,
    exist_ok,
    save_img,
    view_img
):
    # Load model
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz=(640, 640)
    
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    paused=0
    Pitch=0
    yaw=0
    roll=0
    fps = 0.0
    tic = time.time()
    timeo=tic
    # Dataloader
    if webcam:
        print('loading streams:', source)
        dataset = LoadStreams(source, img_size=imgsz)
        bs = 1  # batch_size
    else:
        print('loading images', source)
        dataset = LoadImages(source, img_size=imgsz)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    for path, im, im0s, vid_cap in dataset:
        
        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis= 0)
        else:
            orgimg = im.transpose(1, 2, 0)
        
        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
        '''
這是一行程式碼，用於確認圖像的尺寸是否符合要求。讓我們來解釋一下：

imgsz 是一個變數，用於儲存經過檢查後的圖像尺寸。通常這個尺寸是一個正方形的大小，以便在進行訓練或推理時，可以避免圖像變形。

check_img_size 是一個函數，它接受兩個參數：img_size 和 s。img_size 是預設的圖像尺寸，而 s 是模型中所有層的最大 stride (步長)。這個函數的目的是找到一個符合要求的正方形圖像尺寸，以便將其用於訓練或推理。

model.stride.max() 是獲取模型中所有層的 stride (步長) 的最大值。這個值會被傳遞給 check_img_size 函數，以確保選擇的圖像尺寸不會導致圖像縮小到比 stride (步長) 還小。

最後，確認的圖像尺寸會存儲在 imgsz 變數中，然後可以用於進行訓練或推理。

總結來說，這行程式碼的目的是確保所選擇的圖像尺寸符合模型的要求，並且不會導致圖像被縮小到比 stride (步長) 還小，從而確保模型可以正確處理圖像。
        '''

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()			#這是一個將圖像的通道順序由寬、高、通道（w,h,c）轉換為通道、寬、高（c,w,h）的操作。

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:	#檢查 img 的維度是否為 3（也就是是否是一張單獨的圖像，而不是一個 batch）
            img = img.unsqueeze(0)	#將其轉換成一個包含單個元素的 Tensor（也就是將維度擴展成 [1, c, w, h] 的格式），這樣就形成了一個包含單張圖像的 batch。

        # Inference
        pred = model(img)[0]
        
        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        #print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')
        
        #執行PythonComputerVision-6-CameraCalibration-master可取得相機內參矩陣
        #https://github.com/Nocami/PythonComputerVision-6-CameraCalibration/tree/master
        # 相机内参矩阵
        K = [652.33487191, 0.0, 351.48536718,
        0.0, 648.02978378, 198.83791291,
        0.0, 0.0, 1.0]
        D = [5.03111317e-02, -2.95498746e-01, -1.72208264e-02, -1.95962517e-04, 5.48448668e-01]				#畸变参数，通常是一个1x5的NumPy数组，包括径向畸变和切向畸变等。
        '''
        K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
        0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
        0.0, 0.0, 1.0]
        D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
        '''
        
        
        
        cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)			# 相机内参矩阵（根据你的相机配置提供真实值）
        dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)			# 畸变系数（根据你的相机配置提供真实值）
        #print ("Camera Matrix :\n {0}".format(cam_matrix))
        '''
        object_pts = np.float32([[-4.500000, -4.500000, -2.700000],
                                 [4.500000, -4.500000, -2.700000],
                                 [0.000000, 0.000000, 0.000000],
                                 [-2.700000, 3.000000, -2.500000],
                                 [2.700000, 3.000000, -2.500000]])
                                 

        '''
        object_pts = np.float32([[-53.00000, 45.00000, -37.00000],
                                 [54.00000, 45.00000, -37.00000],
                                 [0.000000, 0.000000, 0.000000],
                                 [-43.00000, -30.00000, -35.00000],
                                 #[37.00000, 30.00000, -35.00000]])
                                 [37.00000, -30.00000, -35.00000],
                                 [-60.00000, 45.00000, -40.00000]])
        

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            
            p = Path(p)  # to Path
            save_path = str(Path(save_dir) / p.name)  # im.jpg

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                #det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()
                det[:, 5:39] = scale_coords_landmarks(img.shape[2:], det[:, 5:39], im0.shape).round()
                
                Lrate=0
                Rrate=0
                         

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = det[j, 5:39].view(-1).tolist()
                    class_num = det[j, 39].cpu().numpy()
                    #print(xyxy,conf,landmarks,class_num)
                    im0 = show_results(im0, xyxy, conf, landmarks, class_num)
                    
                    La=(landmarks[12],landmarks[13])
                    Lb=(landmarks[14],landmarks[15])
                    Lc=(landmarks[18],landmarks[19])
                    Ld=(landmarks[20],landmarks[21])
                    
                    Ra=(landmarks[24],landmarks[25])
                    Rb=(landmarks[26],landmarks[27])
                    Rc=(landmarks[30],landmarks[31])
                    Rd=(landmarks[32],landmarks[33])
                    
                    distance_Lab=cal_distance(La,Lb)
                    distance_Lcd=cal_distance(Lc,Ld)
                    distance_Lad=cal_distance(La,Ld)
                    distance_Lbc=cal_distance(Lb,Lc)
                    
                    distance_LW=distance_Lab+distance_Lcd
                    distance_LH=distance_Lbc+distance_Lad
                    
                    
                    distance_Rab=cal_distance(Ra,Rb)
                    distance_Rcd=cal_distance(Rc,Rd)
                    distance_Rad=cal_distance(Ra,Rd)
                    distance_Rbc=cal_distance(Rb,Rc)
                    


                    distance_RW=distance_Rab+distance_Rcd
                    distance_RH=distance_Rbc+distance_Rad
                    
                    
                    #print(f"distance_LW:{distance_LW}\n")
                    #print(f"distance_LH:{distance_LH}\n")
                    
                    #print(f"distance_RW:{distance_RW}\n")
                    #print(f"distance_RH:{distance_RH}\n")
                    
                    
                    
                    image_pts = np.float32([[landmarks[0], landmarks[1]], [landmarks[2], landmarks[3]], [landmarks[4],landmarks[5]], [landmarks[6],landmarks[7]], [landmarks[8],landmarks[9]], [landmarks[10],landmarks[11]]])
                    #print(image_pts)
                    #print("---------")
                    
                    #相機座標
                    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)	
                    #_, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, None)	
                    #_, rotation_vec, translation_vec = cv2.solvePnPRansac(object_pts, image_pts, cam_matrix, dist_coeffs)
                    #retval, rotation_vec, translation_vec, inliers = cv2.solvePnPRansac(object_pts, image_pts, cam_matrix, dist_coeffs,reprojectionError=0.01, iterationsCount=3000)
                    
                    '''
                    focal_length = det.size()[1]
                    center = (det.size()[1]/2, det.size()[0]/2)
                    camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0, 0, 1]], dtype = "double")
                    
                    print(f"Camera Matrix :\n{camera_matrix}")
                    #print "Camera Matrix :\n {0}".format(camera_matrix)
                    
                    
                    
                    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
                    (success, rotation_vector, translation_vector) = cv2.solvePnP(object_pts, image_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                    print(f"Rotation Vector:\n{rotation_vector}")
                    print(f"Translation Vector :\n{translation_vector}")
                    
                    #print "Rotation Vector:\n {0}".format(rotation_vector)
                    #print "Translation Vector:\n {0}".format(translation_vector)

                    '''
                    
                    
                    
                    #print(retval)
                    #print(rotation_vec)
                    #ro_vec=get_ro_vector(image_pts,object_pts)
                    #print(ro_vec)
                    
                    #get_euler_angle(ro_vec)  
                    #print("-----1-------")
                    #print(translation_vec)
                    #print("-----2-------")
                    #print("rotation_vec:\n",rotation_vec)
                    #print("-----3-------")
                    # 将旋转向量转换为旋转矩阵
                    
                    Lrate=distance_LH/distance_LW
                    Rrate=distance_RH/distance_RW
                    #print(f"Lrate:{Lrate}\n")
                    #print(f"Rrate:{Rrate}\n")
                    
                    
                    _,Pitch,yaw,roll=get_euler_angle(rotation_vec)  
                    #print('Pitch:{}, yaw:{}, roll:{}'.format(Pitch,yaw,roll))
            
            if view_img:
                text = "fps:%d,Pitch:%d,yaw:%d,roll:%d"%(fps,Pitch,yaw, roll)
                cv2.putText(im0, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,(32, 32, 32), 4, cv2.LINE_AA)		#黑底
                cv2.putText(im0, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,(240, 240, 240), 1, cv2.LINE_AA)		#白字
                
                text2 = "Lrate:%.3f,Rrate:%.3f"%(Lrate,Rrate)
                cv2.putText(im0, text2, (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.0,(32, 32, 32), 4, cv2.LINE_AA)		#黑底
                cv2.putText(im0, text2, (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.0,(240, 240, 240), 1, cv2.LINE_AA)		#白字
                
                if Lrate<0.65 or Rrate<0.65:
                    text2 = "Sleep"
                    cv2.putText(im0, text2, (60, 70), cv2.FONT_HERSHEY_PLAIN, 2.0,(32, 32, 32), 4, cv2.LINE_AA)		#黑底
                    cv2.putText(im0, text2, (60, 70), cv2.FONT_HERSHEY_PLAIN, 2.0,(0, 0, 240), 1, cv2.LINE_AA)		#白字
                else:
                    text2 = "Wake Up"
                    cv2.putText(im0, text2, (60, 70), cv2.FONT_HERSHEY_PLAIN, 2.0,(32, 32, 32), 4, cv2.LINE_AA)		#黑底
                    cv2.putText(im0, text2, (60, 70), cv2.FONT_HERSHEY_PLAIN, 2.0,(0, 240, 0), 1, cv2.LINE_AA)		#白字
                    
                if Pitch>3 or 0<yaw<165 or 0>yaw>-160:
                    text2 = "distracted"
                    cv2.putText(im0, text2, (60, 100), cv2.FONT_HERSHEY_PLAIN, 2.0,(32, 32, 32), 4, cv2.LINE_AA)		#黑底
                    cv2.putText(im0, text2, (60, 100), cv2.FONT_HERSHEY_PLAIN, 2.0,(0, 0, 240), 1, cv2.LINE_AA)		#白字
                else:
                    text2 = "concentrate"
                    cv2.putText(im0, text2, (60, 100), cv2.FONT_HERSHEY_PLAIN, 2.0,(32, 32, 32), 4, cv2.LINE_AA)		#黑底
                    cv2.putText(im0, text2, (60, 100), cv2.FONT_HERSHEY_PLAIN, 2.0,(0, 240, 0), 1, cv2.LINE_AA)		#白字
                
                
                
                toc = time.time()
                curr_fps = 1.0 / (toc - tic)
                # calculate an exponentially decaying average of fps number
                fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
                tic = toc
                cv2.imshow('result', im0)
                key = cv2.waitKey(1)
                if key == ord(' '):  # 空格鍵
                    paused = not paused
                    if paused:
                        key =cv2.waitKey(0) 
                        paused=False
                elif key == ord('q'):  # 空格鍵
                    break
                elif key == 27:	#'ESC'
                    break		
                    
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    try:
                        vid_writer[i].write(im0)
                    except Exception as e:
                        print(e)

                    
            
	#python detect_face.py --weights weights/yolov5n-face.pt --view-img
	#python detect_face.py --weights weights/yolov5n-face.pt --source data/pic --img-size 640 --save-img --view-img
	#save to ./runs/detect/exp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5n-face.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--view-img', action='store_true', help='show results')
    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    detect(model, opt.source, device, opt.project, opt.name, opt.exist_ok, opt.save_img, opt.view_img)
