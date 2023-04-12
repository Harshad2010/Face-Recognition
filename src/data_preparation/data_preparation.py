#Import Libraries
import cv2
import sys
import numpy as np
from glob import glob #for list conversation
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from src.logger import logging
from src.exception import FaceRecognitionException

#glob will return all image path in list
fpath = glob('src/data_source/data/female/*.jpg')
mpath = glob('src/data_source/data/male/*.jpg')

class DataPreparation :
    def __init__() :
        pass
    
    def crop_female_images():
        "Crop all female faces from data folder to crop data folder"
        for i in range(len(fpath)):
            try: #inorder to skip runtime error
                
                #if i==5:
                    #break #for testing purpose
                ### Step -1 Read Image and convert to RGB
                # Read Image in BGR format
                img = cv2.imread(fpath[i])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert image fromBGR to RGB
                ### Step -2 Apply Haar Cascade classifier
                haar = cv2.CascadeClassifier("src/model/haarcascade_frontalface_default.xml")
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                face_list = haar.detectMultiScale(gray,1.5,5)
                
                for x,y,w,h in face_list:
                    #cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,255,0),2)
                    #step3 crop face
                    roi = img[y:y+h, x:x+w]
                    #step4 : Save image
                    cv2.imwrite(f"src/data_source/crop_data/female/female_{i}.jpg", roi)
                    print("Image succesfully processed")
            except Exception as e:
                print("Unable to process the image")
                raise FaceRecognitionException(e,sys)
                
    def crop_male_images():
        "Crop all male faces from data folder to crop data folder"
        for i in range(len(mpath)):
            try: #inorder to skip runtime error
                #if i==5:
                    #break #for testing purpose
                ### Step -1 Read Image and convert to RGB
                # Read Image in BGR format
                img = cv2.imread(mpath[i])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert image fromBGR to RGB
                ### Step -2 Apply Haar Cascade classifier
                haar = cv2.CascadeClassifier("src/model/haarcascade_frontalface_default.xml")
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                face_list = haar.detectMultiScale(gray,1.5,5)
                
                for x,y,w,h in face_list:
                    #cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,255,0),2)
                    #step3 crop face
                    roi = img[y:y+h, x:x+w]
                    #step4 : Save image
                    cv2.imwrite(f"src/data_source/crop_data/male/male_{i}.jpg", roi)
                    #print("Image succesfully processed")
            except Exception as e:
                print("Unable to process the image")
                raise FaceRecognitionException(e,sys)
            
        