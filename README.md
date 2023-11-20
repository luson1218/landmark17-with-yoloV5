# landmark17 with yoloV5

facelandmark with yolov5

引言.

將人臉與眼睛特徵（共17點）與yoloV5結合

表現.

檢測方式：
python test_all_pic.py     

shape_predictor_68_face_landmarks.dat 下載處 http://dlib.net/files/

face_landmarks34_230715.pth  可照此網站方式產生 CNN: https://arkalim.org/blog/face-landmarks-detection/

抓到的目標物 / 抓到的人臉數

檢測法   所使用的權重檔（weight）				目標數量 /全部數量  		全部時間        ,            平均一張時間                圖片大小         

All : 					:		1008											      640

CNN(R)face_landmarks34_230715.pth   :		945/1631,Time      160.37259435653687,   	0.159099 sec				    

FPLD(Y)checkpoint.pth.tar  		:		847/1344,Time      44.757962226867676,	0.044402 sec				                    

Dlib(G)shape_predictor_68_face_landmarks.dat:	927/1103,Time      28.865395069122314,   	0.028636 sec				                    

Yolov5_m17(W)mark17_18.pt		:		997/1409,Time      15.387505531311035,    0.015265 sec				                    

Yolov5(W)yolov5s-face.pt		:		1006/1830,Time     13.463255643844604,	0.013356 sec				                    



檢測方式：python test_all_camera.py

影像檔抓到的次數(張數)

AVI  : (28秒左右,每秒30幀) 841(張)

CNN(Red):					465/841		55.29%					

FPLD(Yellow):				509/841		60.52%

Dlib(Green):				422/841		50.18%

Yolov5(White) landmark17:	      724/841		88.23%

Yolov5(W) face 		:		841/841		100.0%

準備訓練資料

執行
python create_labels.py  
產生初始特徵點與圖

使用資料集ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml

https://www.kaggle.com/datasets/toxicloser/ibug-300w-large-face-landmark-dataset

使用yolo5 face 取face 與 5個特徵點 加上Dlib取12個特徵點(眼睛)

產生labels 資料夾 （label.txt 與images）

注意： 由於我們使用原始yoloV5的model 來偵測人臉的部份,所以須先將class Detect(nn.Module)改回原本的型態
      因此需要先將models/yolo.py 里的第一個"class Detect(nn.Module)" mark起來並使用第二個"class Detect(nn.Module)"
      產生label 檔,產生完之後需要在改回來 

Facelandmark_tool        #針對上面產生的資料做細部處理
一些太複雜或不完整的圖也不取用（直接刪除）

Data preparation

此部份的操作可參考https://github.com/deepcam-cn/yolov5-face

Download annotation files from 
https://drive.google.com/file/d/1tU_IjyOwGQfGNUvZGwWWM4SwxKp2PUQ8/view?usp=sharing
只需下載val資料即可

cd data
將之前處理好的labels資料夾放置 data 內

python3 train2yolo.py /labels /save/labels/train  ＃將資料轉成yolo訓練格式（Normalization）

python3 val2yolo.py  /retinaface_gt_v1.1 /save/labels/val

進行訓練
CUDA_VISIBLE_DEVICES="0" python3 train.py --data data/luson_test.yaml --cfg models/yolov5s.yaml --weights weights/yolov5s.pt --epochs 250 --img-size 480

luson_test.yaml   為訓練資料所在路徑

mark17xx.pt		  為訓練得到的weights 檔（mark17_18.pt  為第18次遷移學習 total 9010Epoch(125.63 hr)得到的weights 檔）

進行一些程式修正以detect_m17.py 取代 detect_face

python detect_m17.py --weights weights/mark17_18.pt --view-

引文.

CNN:		https://arkalim.org/blog/face-landmarks-detection/

FPLD:		https://github.com/polarisZhao/PFLD-pytorch

Dlib:		https://github.com/davisking/dlib

Yolov5:		https://github.com/deepcam-cn/yolov5-face

