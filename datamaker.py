import cv2
import time
import numpy as np
import os


image_x, image_y = 64, 64

def create_folder(folder_name):
    if not os.path.exists('./dataset/training_set/' + folder_name):
        os.mkdir('./dataset/training_set/' + folder_name)
    if not os.path.exists('./dataset/test_set/' + folder_name):
        os.mkdir('./dataset/test_set/' + folder_name)
    
        

        
def capture_images(ges_name):
    create_folder(str(ges_name))
    
    cap = cv2.VideoCapture(0)
    f=0
    while(f<100):
        ret, frame = cap.read()
        f+=1
    img_counter = 0
    t_counter = 1
    training_set_image_name = 1
    test_set_image_name = 1

    while(1):

        # reads frames from a camera
        ret, frame = cap.read()
        #optimizing area of interest
        y=len(frame)
        x=len(frame[0])
        x=x-x//6
        y=y-3*y//4
        x_low=x-len(frame[0])//4
        y_high=y+2*len(frame)//5
        area_roi=(x-x_low+29)*(y_high-y+19)
        
        frame=cv2.flip(frame,1)
        kernel=np.ones((3,3),np.uint8)
        # converting BGR to HSV
        roi=frame[y-20:y+2*len(frame)//5, x-15-len(frame[0])//4 :x+15]
        #roi2=roi
        #converting BGR to Grayscale
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        cv2.rectangle(frame,(x_low,y),(x,y_high),(55,175,212),1)
         
        ret,th1 =cv2.threshold(hsv,120,255,cv2.THRESH_TOZERO)
        th1=cv2.cvtColor(th1, cv2.COLOR_BGR2HSV)
        #cv2.imshow('threshold1',th1)

        # 	# define range of skin color in HSV
        lower_red = np.array([0,20,70],dtype=np.uint8)
        upper_red = np.array([20,255,255],dtype=np.uint8)

        # 	# create a skin HSV colour boundary and
        # 	# threshold HSV image
        mask = cv2.inRange(th1, lower_red, upper_red)
        mask= cv2.dilate(mask,kernel,iterations= 4)
        mask= cv2.GaussianBlur(mask,(5,5),100)
        cv2.imshow('threshold2',mask)
        cv2.imshow('orginal',roi)
        res = cv2.bitwise_and(roi,roi,mask=mask)
        #     # Bitwise-AND mask and original image
	#cnts=ut.getContourBiggerThan(contours,minArea=3000,maxArea=40000)
        contours,_ = cv2.findContours(mask,1,2)
        area=0
        for cnt in contours:
            area=area+cv2.contourArea(cnt)
            
        
        if area<area_roi-5000:

            if t_counter <= 350:
                img_name = "./dataset/training_set/" + str(ges_name) + "/{}.png".format(training_set_image_name)
                cv2.imwrite(img_name, roi)
                print("{} written!".format(img_name))
                training_set_image_name += 1


            if t_counter > 350 and t_counter <= 450:
                img_name = "./dataset/test_set/" + str(ges_name) + "/{}.png".format(test_set_image_name)

                cv2.imwrite(img_name, roi)
                print("{} written!".format(img_name))
                test_set_image_name += 1
            t_counter+=1
            if t_counter > 450:
                t_counter = 1
                img_counter += 1
                cap.release()
                cv2.destroyAllWindows()
                break

        if cv2.waitKey(1) == 27:
            break



    cap.release()
    cv2.destroyAllWindows()
    
ges_name = input("Enter gesture name: ")
capture_images(ges_name)
