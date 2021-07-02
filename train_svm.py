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


def convert_to_hog(img):
	edges = cv2.Canny(img, 100, 200)
	edges= cv2.resize(edges,(300,300))
	hog_array=hog(edges,orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=False)
	return hog_array.tolist()
	
	
	
Gesture_Feature=[]
Gesture_Label=[]
Training_Dataset_Folder="./dataset/training_set/"
Gesture_Class=[0,1,2,3,4,5]

for i in Gesture_Class:
	Dir_path=Training_Dataset_Folder+str(i)+"/"
	for filename in os.listdir(Dir_path):
		img=cv2.imread(Dir_path+filename)
		Gesture_Feature.append(convert_to_hog(img))
		Gesture_Label.append(i)
print(len(Gesture_Feature))

OVR_SVM_CLF = OneVsRestClassifier(LinearSVC(random_state=0)).fit(Gesture_Feature, Gesture_Label)

print('Saving SVM model to hog_svm2' +'.pkl')
joblib.dump(OVR_SVM_CLF, 'hog_svm2'+ '.pkl') 
print("Succesfully Trained Model")

#img = cv2.imread('4.png')
#convert_to_hog(img)
