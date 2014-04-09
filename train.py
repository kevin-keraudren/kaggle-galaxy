#!/usr/bin/python

import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt

import sys, os
import cv2
import numpy as np
import scipy.ndimage as nd
import csv
from time import time
import math

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import validation_curve

import joblib
from joblib import Parallel, delayed
from scipy.stats.mstats import mquantiles

import galaxy

def process_galaxy( galaxy_id, transform=0 ):
    #root = "/media/kevin/0026A5FD26A5F3B6/kaggle/galaxy/"
    root = "/vol/biomedic/users/kpk09/kaggle/galaxy/data/"
    f = root + "images_training_rev1/"+galaxy_id+".jpg"
    return galaxy.get_features(f,transform=transform)

def assess(estimator,X,y):
    predictions = estimator.predict(X)
    #print predictions

    #predictions[:,:3] /=2

    scaler1 = joblib.load(galaxy.get_data_folder()+"/scaler1")
    predictions = scaler1.inverse_transform(predictions)
    #predictions *= scale
    
    predictions[predictions<1e-7] = 0

    #y[:,:3] /=2
    y = scaler1.inverse_transform(y)
    #y *= scale
    MSE = (predictions - y)**2
    mse = np.mean(MSE,axis=1)
    rmse = math.sqrt(mse.mean())
    rmse2 = np.sqrt(mse)

    indices = np.argsort(mse)[::-1]
    
    for i in indices[:5]:
        if rmse2[i]>0.1:
            print rmse2[i], int(X[i,0])
        
    return rmse

def shuffle(a, b):
    """
    http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

f = "/vol/biomedic/users/kpk09/kaggle/galaxy/data/training_solutions_rev1.csv"
responses, ids = galaxy.read_responses( f )

scaler1 = StandardScaler()
responses = scaler1.fit_transform(responses)
joblib.dump( scaler1, galaxy.get_data_folder()+"/scaler1" )

# mapping = galaxy.get_classes()
# selection = {}
# for Class in xrange(1,12):
#     classes = np.nonzero(mapping==Class)[0]    
#     for c in classes:
#         q = 0.95
#         threshold = mquantiles( responses[:,c], q )        
#         tmp_selection = np.nonzero(responses[:,c]>=threshold)[0]
#         for i in tmp_selection:
#             selection[i] = 1

# tmp_responses = []
# tmp_ids = []
# for i in selection.keys():
#     tmp_responses.append(responses[i])
#     tmp_ids.append(ids[i])

# responses = np.array(tmp_responses)
# ids = np.array(tmp_ids)

responses,ids = shuffle(responses,np.array(ids))

# responses = responses[:3000]
# ids = ids[:3000]

# standardization: substract the mean then divide by std
#responses -= responses.mean(axis=0)
#scale = responses.max(axis=0)
#responses /= scale
#responses *= 1000


#responses[:,:3] *=2


print "will learn from", len(ids), "galaxies"

all_points = []
all_responses = []
transform = 0
for transform in xrange(2):
    points = Parallel(n_jobs=-1)(delayed(process_galaxy)(galaxy_id,
                                                         transform=transform)
                                 for galaxy_id in ids )
    all_points.extend(points)
    all_responses.extend(responses)

all_points = np.array(all_points)
all_responses = np.array(all_responses)
    
# all_points = points
# all_responses = responses

print "DIM:",len(all_points[0])

print "will train forest with",len(all_points), "points"

scaler2 = StandardScaler()
all_points = scaler2.fit_transform(all_points)

joblib.dump( scaler2, galaxy.get_data_folder()+"/scaler2")

# forest = RandomForestRegressor( n_estimators=100,
#                                 max_depth=10,
#                                 n_jobs=-1 )

# forest = RegressionForest( n_estimators=30,
#                            min_items=5,
#                            max_depth=30,
#                            nb_tests=1000,
#                            test="axis",
#                            verbose=False)
# print forest.get_params()


forest = ExtraTreesRegressor( n_estimators=2000,
                              min_samples_leaf=3,
                              max_depth=60,
                            #  bootstrap=True,
                             n_jobs=-1 )

forest.fit(all_points, all_responses)


#param_name = "max_depth"
#param_range = np.logspace(0, 2, 10)
#param_range = [60]

# param_name = "min_samples_leaf"
# param_range = np.logspace(0, 2, 5)

#param_name = "bootstrap"
#param_range = np.logspace(-1, 0, 10)

# param_name = "max_features"
# param_range = np.logspace(-1, 0, 10)

#forest = GradientBoostingRegressor(loss='lad')
#param_name = "n_estimators"
#param_range = np.logspace(0, 2, 5).astype('int')
#param_name = "max_depth"
#param_range = np.logspace(0, 2, 10)

# forest = SVR()
# param_name = "gamma"
# param_range = np.logspace(-6, -1, 5)

# all_points = np.concatenate( (np.reshape(np.array(ids,dtype='float'),(len(ids),1)), all_points),axis=1 )

# train_scores, test_scores = validation_curve(
#     forest, all_points, all_responses, param_name=param_name, param_range=param_range,
#     cv=10, scoring=assess, n_jobs=-1)#,verbose=False)

# print train_scores, test_scores
# print train_scores.mean(), test_scores.mean()

# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# plt.title("Validation Curve with Random Forest")
# plt.xlabel(param_name)
# plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2, color="r")
# plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
#              color="g")
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2, color="g")
# plt.legend(loc="best")
# plt.savefig(param_name+".png")

# exit(0)

#param_grid = {"max_depth": [20,30,None],
#              "min_samples_leaf": [1, 3, 5, 10],
#              "bootstrap": [True, False]}

# run grid search
#grid_search = GridSearchCV(forest, param_grid=param_grid)
#start = time()
#grid_search.fit(X, y)

#print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#      % (time() - start, len(grid_search.grid_scores_)))

#print grid_search.best_score_,grid_search.best_params_

#forest = grid_search.best_estimator_

if not os.path.exists(galaxy.get_data_folder()):
    os.makedirs(galaxy.get_data_folder())

joblib.dump(forest, galaxy.get_data_folder()+"/galaxy_forest")

# importances = forest.feature_importances_
# indices = np.argsort(importances)[::-1]

# # Print the feature ranking
# print("Feature ranking:")

# for f in range(10):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))    
