# Hand_Gesture_Recognition

This project is made by me under the guidance of Professor Souvik Chakraborty, IIT Delhi.
In this project, we made a classifier based on <b>SVM(Support Vector Machine></b> to differentiate between different hand gestures.
We are differentiating between 6 different gestures labelled [0,1,2,3,4,5], and predicts gestures with overall accuracy of <b>87.66%</b>. This best part of our model is:- It also <b>works for real time inputs</b>.


<h3>Demonstration</h3>


https://user-images.githubusercontent.com/58558528/124331174-82306680-dbac-11eb-98b9-7bd085ad5c05.mp4



<h3>Methodology</h3>
This model is taking input from the webcam of your computer and then at each timestep:-
<ol>
  <li>It <b>pre-process the image</b> to remove noises and crop image to take only region which is of interest to us.
  <li>Detect edges of hand by using <b>Canny's Edge Detection</b>.
  <li>Apply <b>HOG(Histogram Of Oriented Gradients)</b> to extract useful features from edge detected image.
  <li>Use extracted features to predict gesture by <b>trained SVM model</b>.
</ol>
<h3>Prediction Scores</h3>
<table>
  <th>Gesture Label</th>
  <th>Accuracy of predicting correctly in 100 test samples.</th>
    <tr  align="center"><td>0</td> <td>76%</td></tr>
    <tr  align="center"><td>1</td> <td>93%</td></tr>
    <tr  align="center"><td>2</td> <td>97%</td></tr>
    <tr  align="center"><td>3</td> <td>79%</td></tr>
    <tr  align="center"><td>4</td> <td>100%</td></tr>
    <tr  align="center"><td>5</td> <td>81%</td></tr>
  <tr  align="center"><td><b>Overall</b></td><td><b>87.66%</b></td></tr>
</table>

To run the model you have to run "Final.py" file.<br><br>
But before running make sure you have downloaded all the files that is included in this folder-<br>
"datamaker.py", "dataset.zip", "hog_svm2.pkl", "train_svm.py", "Final.py", "test_svm.py","dataset.zip".
<br><br>
Also make sure that you have installed all the libraries required to run the code some of the libraries are listed below:-<br>
OpenCv
,Matplotlib
,sklearn
,joblib
<br><br>
For using our formed dataset extract "dataset.zip".<br><br>
For making your own dataset:-<br>
Run 'datamaker.py' and enter label for data you want to create database. Then show corresponding gesture in rectangle area 
<br><br>
For training the model:-<br>
Run 'train_svm.py"
<br><br>
For testing the model:-<br>
Run "test_svm.py"
<br><br>
For running real time gesture classifier:-<br>
Run "Final.py"
<br><br>
Final Report, presentation video, presentation slides are included in report folder.

<h2>Thank You</h2>

