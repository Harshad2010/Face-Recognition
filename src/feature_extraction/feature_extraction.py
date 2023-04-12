import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from src.logger import logging
from src.exception import FaceRecognitionException

## Eigen face
from sklearn.decomposition import PCA
import pickle

def feature_extraction():
    try:    
        # Load the data
        data = pickle.load(open('src/data_source/data/data_images_100_100.pickle', mode='rb'))

        #### Eigen face
        # Mean Face
        X= data.drop("gender",axis=1).values #all images
        mean_face = X.mean(axis=0) #flatten mean face

        #Subtract data with mean face
        # transform data
        X_t = X - mean_face

        pca = PCA(n_components=None, whiten=True, svd_solver='auto')
        pca.fit(X_t)

        exp_var_df = pd.DataFrame()
        exp_var_df['explained_var'] = pca.explained_variance_ratio_

        exp_var_df['cum_explained_var'] =exp_var_df['explained_var'].cumsum()

        exp_var_df['principal_components'] = np.arange(1, len(exp_var_df)+1)
        #exp_var_df.head()

        pca_50 = PCA(n_components=50, whiten=True, svd_solver='auto')
        pca_data = pca_50.fit_transform(X_t) #X_t original data subtracted by the mean face

        y = data['gender'].values #independent variables

        # Saving data and models
        np.savez('src/data_source/data/data_pca_50_target', pca_data,y)

        # Saving the model , dict convient way to save the mode
        pca_dict = {'pca': pca_50, 'mean_face':mean_face}

        pickle.dump(pca_dict, open('src/model/pca_dict.pickle','wb'))
        
    except Exception as e:
            raise FaceRecognitionException(e,sys)