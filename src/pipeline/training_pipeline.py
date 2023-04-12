import numpy as np
import pandas as pd
import sklearn
import pickle

import matplotlib.pyplot as plt
import cv2
from src.logger import logging
from src.exception import FaceRecognitionException

# Load all models
haar = cv2.CascadeClassifier("src/model/haarcascade_frontalface_default.xml") #cascade classifier
model_svm = pickle.load(open("src/model/model_svm.pickle", mode="rb")) #machine learning model (SVM)
pca_models = pickle.load(open("src/model/pca_dict.pickle", mode="rb")) #pca dictionary

model_pca = pca_models['pca'] #PCA model
mean_face_arr = pca_models['mean_face'] # Mean face

def training_pipeline():
    try:
         # step-01: read image
        img = cv2.imread("getty_test.jpg")
        # step-02: convert into gray scale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # step-03: crop the face (using haar cascade classifier)
        faces = haar.detectMultiScale(gray,1.5,3)
        predictions= []
        for x,y,w,h in faces :
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            roi = gray[y:y+h, x:x+w]
            # step-04: normalization (0-1)
            roi = roi/255.0
            # step-05: resize images (100,100)
            if roi.shape[1]>100:
                roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
            else:
                roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
            # step-06: Flattening (1x10000)
            roi_reshape= roi_resize.reshape(1,10000)
            # step-07: subtract with mean
            roi_mean = roi_reshape - mean_face_arr 
            # step-08: get eigen image (apply roi_mean to pca)---------NOT UNDERSTOOD
            eigen_image= model_pca.transform(roi_mean)
            # step-09: Eigen Image for visualization---------NOT UNDERSTOOD
            eig_img = model_pca.inverse_transform(eigen_image)
            # step-10: pass to ml model(svm) and get prediction.
            results= model_svm.predict(eigen_image)
            prob_score= model_svm.predict_proba(eigen_image)
            prob_score_max= prob_score.max()
            
            # step-10: generate report
            text= "%s : %d"%(results, prob_score_max*100)
            # defining the color based on results
            if results[0]=='male':
                color=(255,255,0)
            else:
                color=(255,0,255)
                
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.rectangle(img,(x,y-30),(x+w,y),color,-1)
            cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,2.5,(255,255,255),5) #---------NOT UNDERSTOOD
            output = {
                'roi' : roi,
                'eig_img' :eig_img,
                'prediction_name': results[0],
                'score': prob_score_max
            }
            predictions.append(output)#list of output dictionaries which 4elements in each dictionary
    
    except Exception as e:
            raise FaceRecognitionException(e,sys)
                