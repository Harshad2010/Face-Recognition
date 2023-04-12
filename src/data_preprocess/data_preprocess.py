# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 #Computer vision library
from glob import glob
import pickle
import sys
from src.logger import logging
from src.exception import FaceRecognitionException

# extract path of male and female in crop_data folder and put them in list.
class DataPreprocess :
    def __init__() :
        pass
    
    def preprocess():
        "Preprocesses all data"
        try:
            fpath = glob("src/data_source/crop_data/female/*.jpg")
            mpath = glob("src/data_source/crop_data/male/*.jpg")
            
            # Create the df
            df_female = pd.DataFrame(fpath,columns=['filepath'] )
            df_female['gender']= 'female'
            df_male = pd.DataFrame(mpath, columns= ['filepath'])
            df_male['gender']= 'male'
            df= pd.concat((df_female, df_male), axis=0) #axis=0- concatenation would take place in rows explicitly
            df_filter['data'] = df_filter['filepath'].apply(DataPreprocess.structuring)
            data = df_filter['data'].apply(pd.Series)  
            data.columns = [f'pixel_{i}' for i in data.columns]
            
            ### Data Normalization
            # dividing the value by maxm value which is 255
            data= data/255.0
            data['gender']= df_filter['gender']
            pickle.dump(data, open('src/data_source/data/data_images_100_100.pickle', mode="wb"))
            df['dimension']= df['filepath'].apply(DataPreprocess.get_size)
            df_filter = df.query('dimension > 60')
            
        except Exception as e:
            raise FaceRecognitionException(e,sys)
    
    @classmethod
    def get_size(path):
        try:
            img = cv2.imread(path)
            return img.shape[0]
        except Exception as e:
            raise FaceRecognitionException(e,sys)
            
    @classmethod
    def structuring(path):
        " structuring function converts Unstructured data into Structured data"
        try:
            # step 1
            img =cv2.imread(path) #BGR
            
            # step-2 : convert to grayscale
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            #step 3 : resize to 100x100 array
            size= gray.shape[0]
            if size>100:
                # cv2.INTER_AREA (SHRINK)
                gray_resize = cv2.resize(gray,(100,100),cv2.INTER_AREA)
            else:
                # cv2.INTER_CUBIC (ENLARGE)
                gray_resize = cv2.resize(gray,(100,100),cv2.INTER_CUBIC)

            #step4 : Flatten image (1x10,000)
            flatten_image = gray_resize.flatten()
            return flatten_image
        
        except:
            return None
    
