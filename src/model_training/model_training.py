import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pickle
from src.logger import logging
from src.exception import FaceRecognitionException


class ModelTrainer:
    
    def __init__() :
        pass
    
    def model_trainer():
        try:
            # Load numpy array
            data= np.load('src/data_source/data/data_pca_50_target.npz')
            data.files

            #['arr_0', 'arr_1'] :[PCA, TARGETVARIABLE]
            data.allow_pickle = True

            x = data['arr_0'] #pca data with 50 components
            y= data['arr_1'] #target or dependent variable

            ## SPlit the data into train and test
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, stratify=y)

            # Training ML model
            model_svc = SVC(probability=True)

            param_grid = {'C':[0.5,1,10,20,30,50],
                        'kernel': ['rbf', 'poly'],
                        "gamma": [0.1,0.05,0.01,0.001,0.002,0.005],
                        "coef0":[0,1]}

            model_grid = GridSearchCV(model_svc, 
                                    param_grid=param_grid, 
                                    scoring='accuracy',
                                    cv=3,
                                    verbose=2)

            model_grid.fit(x_train, y_train)
            model_final = model_grid.best_estimator_
            y_pred = model_final.predict(x_test) #predicted values
            
            cr = metrics.classification_report(y_test,y_pred, output_dict=True)
            kp_score = metrics.cohen_kappa_score(y_test,y_pred)
            roc_auc_score = metrics.roc_auc_score(np.where(y_test== "male", 1,0),
                                                    np.where(y_pred=="male",1,0))
            pickle.dump(model_final, open("src/model/model_svm.pickle", mode="wb"))
            
        except Exception as e:
            raise FaceRecognitionException(e,sys)
            
        
            

