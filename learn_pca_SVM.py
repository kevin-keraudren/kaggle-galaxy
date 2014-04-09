#!/usr/bin/python

from time import time
import logging
import pylab as pl

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import sys, os
import cv2
import numpy as np
import scipy.ndimage as nd
import csv

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

import joblib
from joblib import Parallel, delayed

import galaxy

import logging
from time import time

from numpy.random import RandomState

from sklearn import decomposition

from scipy.stats.mstats import mquantiles

thumbnail = "img"

def process_galaxy( galaxy_id, transform=0 ):
    #root = "/media/kevin/0026A5FD26A5F3B6/kaggle/galaxy/"
    root = "/vol/biomedic/users/kpk09/kaggle/galaxy/data/"
    f = root + "images_training_rev1/"+galaxy_id+".jpg"
    if thumbnail=="img":
        return galaxy.get_features(f,tiny_img=True,transform=transform)
    elif thumbnail=="grad":
        return galaxy.get_features(f,tiny_grad=True,transform=transform)
    
f = "/vol/biomedic/users/kpk09/kaggle/galaxy/data/training_solutions_rev1.csv"
#f = "/media/kevin/0026A5FD26A5F3B6/kaggle/galaxy/training_solutions_rev1.csv"
responses, ids = galaxy.read_responses( f )

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
mapping = galaxy.get_classes()

for Class in xrange(1,12):
    classes = np.nonzero(mapping==Class)[0]

    X = []
    Y = []
    svm_class = 0
    
    for c in classes:
        q = 0.95
        if Class == 8 or Class == 11:
            q = 0.98
        threshold = mquantiles( responses[:,c], q )
        
        selection = np.nonzero(responses[:,c]>=threshold)[0]

        selected_ids = []

        for i in selection:
            selected_ids.append(ids[i])

        print "will learn from", len(selected_ids), "galaxies for Class", Class,c,svm_class

        for transform in [0]:#xrange(3):
            points = Parallel(n_jobs=-1)(delayed(process_galaxy)(galaxy_id,transform=transform)
                                         for galaxy_id in selected_ids )
        X.extend(points)
        Y.extend([svm_class for i in xrange(len(points))])
        svm_class += 1

    X = np.array(X,dtype='float')
    Y = np.array(Y,dtype='int')
        
    print "got",len(X),"points"
    print np.bincount(Y,minlength=svm_class)             

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 50

    print "Extracting the top %d eigenfaces from %d faces" % (n_components, X.shape[0])
    t0 = time()
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X)
    print "done in %0.3fs" % (time() - t0)

    print pca.explained_variance_ratio_

    print "Projecting the input data on the eigenfaces orthonormal basis"
    t0 = time()
    X_pca = pca.transform(X)
    print "done in %0.3fs" % (time() - t0)

    # save PCA
    joblib.dump(pca, galaxy.get_data_folder()+"/pca_"+thumbnail+"_Class"+ str(Class)+"_")

    # train best SVM
    clf = SVC( kernel='rbf',
               class_weight='auto',
               probability=True,
               C=50000.0,
               gamma=5e-05 )
    clf.fit(X_pca, Y)
    joblib.dump(clf, galaxy.get_data_folder()+"/pca_"+thumbnail+"_SVM_Class" + str(Class)+"_")

exit(0)

###############################################################################
# Train a SVM classification model

print"Fitting the classifier to the training set"
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.00001, 0.00005,0.0001, 0.0005, 0.001], }
clf = GridSearchCV( SVC(kernel='rbf',
                        class_weight='auto'),
                    param_grid,
                    n_jobs=-1)
clf = clf.fit(X_pca, Y)
print "done in %0.3fs" % (time() - t0)
print "Best estimator found by grid search:"
print clf.best_estimator_
print clf.best_score_
print clf.best_params_
