import cv2
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


OVR_SVM_CLF = joblib.load('hog_svm2.pkl')

def convert_to_hog(img):
	edges = cv2.Canny(img, 100, 200)
	edges= cv2.resize(edges,(300,300))
	hog_array=hog(edges,orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=False)
	return np.array(hog_array).reshape((1,len(hog_array)))
	
Testing_Dataset_Folder="./dataset/test_set/"

Gesture_Class=[0,1,2,3,4,5]

accuracy=0
for i in Gesture_Class:
	Dir_path=Testing_Dataset_Folder+str(i)+"/"
	scorei=0
	for filename in os.listdir(Dir_path):
		img=cv2.imread(Dir_path+filename)
		temp=convert_to_hog(img)
		predictionFrame = OVR_SVM_CLF.predict(temp)[0]
		if(predictionFrame==i):
			scorei+=1
	accuracy+=scorei
	print("Accuracy for predicting "+str(i)+" is :"+str(scorei)+"%")
	
print("Overall Accuracy : "+str(accuracy/6)+"%")
