#!/usr/bin/python

import sys, os
import cv2
import numpy as np
import scipy.ndimage as nd
import csv
import math

import joblib
from joblib import Parallel, delayed

import galaxy

def to_dict(responses,id_responses):
    res = {}
    for r,i in zip(responses,id_responses):
        res[i] = r
    return res

responses,id_responses = galaxy.read_responses("/vol/biomedic/users/kpk09/kaggle/galaxy/data/training_solutions_rev1.csv" )

predictions,id_predictions = galaxy.read_responses(sys.argv[1])

ground_truth = to_dict(responses,id_responses)
MSE = np.zeros( (len(id_predictions),37), dtype="float" )
n = 0
for p,i in zip(predictions,id_predictions):
    MSE[n] = (p - ground_truth[i])**2
    n += 1
    
mse = np.mean(MSE,axis=0).mean()
print "MSE:", mse
print "RMSE:", math.sqrt(mse)

scores = MSE.mean(axis=1)
rank = np.argsort(scores)[::-1]

for i in xrange(20):
    n = rank[i]
    galaxy = id_predictions[n]
    print scores[n], galaxy+".jpg", responses[n,:3],predictions[n,:3]
