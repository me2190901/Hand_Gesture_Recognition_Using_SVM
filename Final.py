import cv2
import math
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import numpy as np
import os
from sklearn.svm import LinearSVC
import sklearn.svm as svm
from sklearn.multiclass import OneVsRestClassifier
import joblib

#Loading SVM model
OVR_SVM_CLF = joblib.load('hog_svm2.pkl')

# capture frames from a camera
cap = cv2.VideoCapture(0)

f=0
while(f<100):
	ret, frame = cap.read()
	f+=1

# loop runs if capturing has been initialized
i=0
while(1):
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break
		
	# reads frames from a camera
	ret, frame = cap.read()
	#optimizing area of interest
	y=len(frame)
	x=len(frame[0])
	x=x-x//6
	y=y-3*y//4
	x_low=x-len(frame[0])//4
	y_high=y+2*len(frame)//5
	
	frame=cv2.flip(frame,1)
	#Setting Kernal variables
	kernel=np.ones((3,3),np.uint8)
	#Setting region of interest
	roi=frame[y-20:y_high, x_low-15 :x+15]
	#Setting method for area of contour
	roi2=roi.copy()
	roi2=cv2.resize(roi2,(300,300))
	area_roi=(len(roi2)-1)*(len(roi2[0])-1)
	
	
	#converting BGR to Grayscale and HSV
	hsv = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
	roi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	roi2= cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
	
	ret,th1 =cv2.threshold(hsv,120,255,cv2.THRESH_TOZERO)
	th1=cv2.cvtColor(th1, cv2.COLOR_BGR2HSV)
 	
# 	# define range of skin color in HSV
	lower_red = np.array([0,20,70],dtype=np.uint8)
	upper_red = np.array([20,255,255],dtype=np.uint8)
	
# 	# create a skin HSV colour boundary and
# 	# threshold HSV image
	mask = cv2.inRange(th1, lower_red, upper_red)
	mask= cv2.dilate(mask,kernel,iterations= 4)
	mask= cv2.GaussianBlur(mask,(5,5),100)
	
	
	contours,hierarchy = cv2.findContours(mask,1, 2)
	area=0
	for cnt in contours:
		area=area+cv2.contourArea(cnt)
	
	#Detecting Edges of Image
	edges = cv2.Canny(roi,100,200)
	edges= cv2.resize(edges,(300,300))
	# Display edges in a frame
	cv2.imshow('Edges',edges)
	# Setting rectangle on full area to specify area of interest.
	cv2.rectangle(frame,(x_low,y),(x,y_high),(255,0,0),1)
	
	
	
	text="Object Not Detected"
	#print(str(area_roi)+" "+str(area))
	if area-area_roi:
		i+=1
		hog_array=hog(edges,orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=False)
		hog_list=np.array(hog_array).reshape((1,len(hog_array)))
		#Predicting Gesture
		predictionFrame = OVR_SVM_CLF.predict(hog_list)[0]
		text=str(predictionFrame)
	fontScale = 1.5
	font = cv2.FONT_HERSHEY_SIMPLEX
	# Red color in BGR
	color =(0,0,255)
	
	# Line thickness of 2 px
	thickness = 2
	#Putting Predicted Text Gesture in Frame
	cv2.putText(frame,text,(50,70),font, fontScale, color, thickness, cv2.LINE_AA)
                   
	# Display an original image
	cv2.imshow('Original',frame)
	
	


# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
