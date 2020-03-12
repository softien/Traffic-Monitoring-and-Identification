# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:01:06 2019

@author: Dilawar ALI
"""


import numpy as np
import argparse
import imutils
import time
import cv2
import os
import xlwt 
from xlwt import Workbook

from findcolor import *
from getCoord import *
#from matchScore import *
from imageMatch import *

inc = 0;
# Workbook is created 
wb = Workbook() 
  
# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 
args = {}

# Input Files
args['input'] = 'videos/topview5.mp4'
args['output']  = 'output/try2.avi'
args['yolo'] = 'yolo-coco'
args['confidence'] = 0.5
args['threshold'] = 0.3

# Output Files
filesave = 'try.xls'
#args = vars(ap.parse_args())

flag = 0
regID = 0
carFound = []

                      
# Model and Weights      
# load the COCO class labels our YOLO model was trained on
ch = input('Enter Video Type: 1. Side -- 2. Aerial : ')
if ch == "1":
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
else:
    labelsPath = os.path.sep.join([args["yolo"], "aerial.names"])
    weightsPath = os.path.sep.join([args["yolo"], "yolov3-aerial.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolo_aerial_cfg.cfg"])

LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
colorCar = np.random.randint(0,255,(2000,3))
# derive the paths to the YOLO weights and model configuration
#weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
#configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

#weightsPath = os.path.sep.join([args["yolo"], "yolov3-aerial.weights"])
#configPath = os.path.sep.join([args["yolo"], "yolo_aerial_cfg.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
fps = vs.get(cv2.CAP_PROP_FPS)
# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1
FN = 0
X = []
Y = []
prevFrame = []
# loop over frames from the video file stream
while True:
    FN = FN + 1
    print('FN', FN)
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    frameCopy = frame.copy()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# if the frame was not grabbed, then we have reached the end
	# of the stream
    if not grabbed:
        break

	# if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
    
    blob = cv2.dnn.blobFromImage(frame, 1/255 , (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
    cl = []
    indexXY = []
    dim = []
    boxes = []
    confidences = []
    classIDs = []

	# loop over each of the layer outputs
    for output in layerOutputs:
		# loop over each of the detections
        for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
            if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
                x = int(centerX)
                y = int(centerY)

				# update our list of bounding box coordinates,
				# confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

	# ensure at least one detection exists
    if len(idxs) > 0:
		# loop over the indexes we are keeping
        for i in idxs.flatten():
			# extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            if flag == 0:               
                [x1, y1] = getCoord(frame)
                [x2, y2] = getCoord(frame)
                
                flag = 1
                
            if (((x > x1) & (x < x2)) & ((y > y1) & (y < y2))): 
                Cx = int(x - (w / 2))
                Cy = int(y - (h / 2))
                
                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                
                rectSize = 3
                imgPart = frame[Cy:Cy+h, Cx:Cx+w]

                findcl = findcolor(imgPart)
                cl.append([findcl[0], findcl[1], findcl[2]])
    
                indexXY.append([x,y])
                dim.append([width, height])
                    
                c1 = findcl[0]
                c2 = findcl[1]
                c3 = findcl[2]
                    
                if FN == 1:
                    regID +=1
                    carFound.append([regID, [x, y, w, h, c1, c2, c3, FN]])
                    text = "{}".format(regID)
                    
                else:
                    if FN == 115:
                        print('115')
                    Found = 0
                    lstNew = [x, y, w, h, int(c1), int(c2), int(c3), FN]
                    
                    sumDiff = 1000
                    probRegID = 0
                    
                    if w > h:
                        Limit = w
                    else:
                        Limit = h
                        
                    Found = 0
                    lstNew = [x, y, w, h, int(c1), int(c2), int(c3)]
                    
                    sumDiff = 1000
                    probRegID = 0
                    trendX = []
                    trendY = []
                    trendW = []
                    trendH = []
                    
                    trendC1 = []
                    trendC2 = []
                    trendC3 = []
                    frameTrnd = []

                    for xi in range(len(carFound)):
                        track = carFound[xi][1:]
                        for lst in track:
                            trendX.append(lst[:][0])
                            trendY.append(lst[:][1])
                            trendW.append(lst[:][2])
                            trendH.append(lst[:][3])
                            
                            trendC1.append(lst[:][4])
                            trendC2.append(lst[:][5])
                            trendC3.append(lst[:][6])
                            frameTrnd.append(lst[:][7])
                            
                        
                        firstX = trendX[0]
                        firstY = trendY[0]
                        firstW = trendW[0]
                        firstH = trendH[0]
                        firstC1 = trendC1[0]
                        firstC2 = trendC2[0]
                        firstC3 = trendC3[0]
                            
                        prevX = trendX[len(trendX)-1]
                        prevY = trendY[len(trendY)-1]
                        prevW = trendW[len(trendW)-1]
                        prevH = trendH[len(trendH)-1]
                        prevC1 = trendC1[len(trendC1)-1]
                        prevC2 = trendC2[len(trendC2)-1]
                        prevC3 = trendC3[len(trendC3)-1]
                        lstFrameNo = frameTrnd[len(frameTrnd)-1]
                        
                        lstPrev = [prevX, prevY, prevW, prevH, int(prevC1), int(prevC2), int(prevC3)]
                        
                        diff = [abs(a_i - b_i) for a_i, b_i in zip(lstPrev, lstNew)]
                        tot = (diff[0] + diff[1])
                        
                        if (abs(prevX - firstX) > abs(prevY - firstY)):
                            Limit = w + 5
                        else:
                            Limit = h + 5
                        if (lstFrameNo > (FN-fps)):
                            frameDiff = FN - lstFrameNo
                            if frameDiff != 0:
                                if  (tot < sumDiff):
                                    if (((abs(x-prevX) < (Limit)) & ((abs(y-prevY) < (Limit))))):
                                        sumDiff = tot
                                        probRegID = xi
                                        Found = 1
                   
                    if Found == 0:
                        sumDiff = 1000
                        for xi in range(len(carFound)):
                            trendX = []
                            trendY = []
                            trendW = []
                            trendH = []
                            
                            trendC1 = []
                            trendC2 = []
                            trendC3 = []
                            frameTrnd = []
                            track = carFound[xi][1:]
                            for lst in track:
                                trendX.append(lst[:][0])
                                trendY.append(lst[:][1])
                                trendW.append(lst[:][2])
                                trendH.append(lst[:][3])
                                
                                trendC1.append(lst[:][4])
                                trendC2.append(lst[:][5])
                                trendC3.append(lst[:][6])
                                frameTrnd.append(lst[:][7])
                                
                            prevX = trendX[len(trendX)-1]
                            prevY = trendY[len(trendY)-1]
                            prevW = trendW[len(trendW)-1]
                            prevH = trendH[len(trendH)-1]
                            prevC1 = trendC1[len(trendC1)-1]
                            prevC2 = trendC2[len(trendC2)-1]
                            prevC3 = trendC3[len(trendC3)-1]
                            lstFrameNo = frameTrnd[len(frameTrnd)-1]
                            
                            if (lstFrameNo > (FN-fps)):
                                lstPrev = [prevX, prevY, prevW, prevH, int(prevC1), int(prevC2), int(prevC3), lstFrameNo]
                                diff = [abs(a_i - b_i) for a_i, b_i in zip(lstPrev, lstNew)]
                                tot = (diff[0] + diff[1])
                                colScore = (diff[4] + diff[5] + diff[6])
                                frameDiff = FN - lstFrameNo
                                if frameDiff > 0:
                                    if  (tot < sumDiff):
                                        if ((x > abs(prevX - (prevW/2))) & (x < abs(prevX + (prevW/2))) & (y > abs(prevY - (prevH/2))) & (y < abs(prevY + (prevH/2)))):
                                            sumDiff = tot
                                            probRegID = xi
                                            Found = 1
                                            
                    if Found == 0:     
                        for xi in range(len(carFound)):
                            trendX = []
                            trendY = []
                            trendW = []
                            trendH = []
                            
                            trendC1 = []
                            trendC2 = []
                            trendC3 = []
                            frameTrnd = []
                            track = carFound[xi][1:]
                            for lst in track:
                                trendX.append(lst[:][0])
                                trendY.append(lst[:][1])
                                trendW.append(lst[:][2])
                                trendH.append(lst[:][3])
                                
                                trendC1.append(lst[:][4])
                                trendC2.append(lst[:][5])
                                trendC3.append(lst[:][6])
                                frameTrnd.append(lst[:][7])
                            
                            firstX = trendX[0]
                            firstY = trendY[0]
                            firstW = trendW[0]
                            firstH = trendH[0]
                            firstC1 = trendC1[0]
                            firstC2 = trendC2[0]
                            firstC3 = trendC3[0]
                            
                            prevX = trendX[len(trendX)-1]
                            prevY = trendY[len(trendY)-1]
                            prevW = trendW[len(trendW)-1]
                            prevH = trendH[len(trendH)-1]
                            prevC1 = trendC1[len(trendC1)-1]
                            prevC2 = trendC2[len(trendC2)-1]
                            prevC3 = trendC3[len(trendC3)-1]
                            lstFrameNo = frameTrnd[len(frameTrnd)-1]
                            
                            if (lstFrameNo > (FN-fps)):
                                lstPrev = [prevX, prevY, prevW, prevH, int(prevC1), int(prevC2), int(prevC3), lstFrameNo]
                                diff = [abs(a_i - b_i) for a_i, b_i in zip(lstPrev, lstNew)]
                                
                                tot = (diff[0] + diff[1])
                                colScore = (diff[4] + diff[5] + diff[6])
                                frameDiff = FN - lstFrameNo
                                
                                diffF = [frameTrnd[i+1]-frameTrnd[i] for i in range(len(frameTrnd)-1)]
                                if (diffF != []) & (frameDiff > 0):
                                    diffX = [trendX[i+1]-trendX[i] for i in range(len(trendX)-1)]
                                    realDiffX = [ri/rj for ri, rj in zip(diffX, diffF)]
                                    avgX = sum(realDiffX)/len(realDiffX)
                                    
                                    diffY = [trendY[i+1]-trendY[i] for i in range(len(trendY)-1)]
                                    realDiffY = [ri/rj for ri, rj in zip(diffY, diffF)]
                                    avgY = sum(realDiffY)/len(realDiffY)
                                    
                                    if ((abs((x-prevX)/frameDiff - (avgX)) < 5) & (abs((y-prevY)/frameDiff - (avgY)) < 5)):
                                        img1 = imgPart
                                        PrevCx = int(prevX - (prevW / 2))
                                        PrevCy = int(prevY - (prevH / 2))
                                        img2 = prevFrame[PrevCy:PrevCy+prevH, PrevCx:PrevCx+prevW]
                                        
                                        score = imageMatch(img1, img2)
                                        print('Score: ', score)
                                        
                                        if score > 20:
                                            if (diff[4] < 20 & diff[5] < 20 & diff[6] < 20):
                                                sumDiff = tot
                                                probRegID = xi
                                                Found = 1

                    
                    if Found == 1:
                        carFound[probRegID].append([x, y, w, h, int(c1), int(c2), int(c3), FN])
#                        text = "{} {}: {:.4f}".format(probRegID, LABELS[classIDs[i]], confidences[i])
                        text = "{}".format(probRegID+1)
                    else:
                        regID +=1
                        carFound.append([regID, [x, y, w, h, int(c1), int(c2), int(c3), FN]])
#                        text = "{} {}: {:.4f}".format(regID, LABELS[classIDs[i]], confidences[i])
                        text = "{}".format(regID)

        			#if round(x+(w/2)) > 0 & round(y+(h/2)) > 0:
                    cv2.circle(frame, (x, y), 8, color, -1)
                    
        			#cv2.line(frame, (x_prev,y_prev),(x,y), color, 2)
                    x_prev = x
                    y_prev = y
                    
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

	# check if the video writer is None
    
    for car in range(len(carFound)):
        pathX = []
        pathY = []
        trackCar = carFound[car][1:]
        for tk in trackCar:
            if ((FN - tk[:][7]) < 14) :
                pathX.append(tk[:][0])
                pathY.append(tk[:][1])
        
        for pth in range(len(pathX)):
            cv2.circle(frame, (pathX[pth], pathY[pth]), round(pth/2), colorCar[car].tolist(), -1)
      
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
    
    if writer is None:
		# initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, fps, (frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

	# write the output frame to disk
    writer.write(frame)
    prevFrame = frameCopy
#    [x3, y3] = getCoord(frameCopy)
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()