#!/usr/bin/python

import matplotlib
matplotlib.use('cairo')

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
import pylab as pl

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition

thumbnail = "img"

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
n_row, n_col = 6,6
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)

###############################################################################
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    pl.figure(figsize=(2. * n_col, 2.26 * n_row))
    pl.suptitle(title, size=16)
    for i, comp in enumerate(images):
        pl.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        pl.imshow(comp.reshape(image_shape), cmap=pl.cm.gray,
                  interpolation='nearest',
                  vmin=-vmax, vmax=vmax)
        pl.xticks(())
        pl.yticks(())
    pl.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    pl.savefig(title+'.png')
    
###############################################################################
# Plot a sample of the input data

for i in xrange(1,12):
    #pca = joblib.load(galaxy.get_data_folder()+"/pca_"+str(i)+"_")
    pca = joblib.load(galaxy.get_data_folder()+"/pca_"+thumbnail+"_Class"+ str(i)+"_")
    print pca.components_.shape
    print pca.explained_variance_ratio_
    plot_gallery("Class_"+thumbnail+str(i), pca.components_[:n_components])

    cv2.imwrite("mean_Class_"+thumbnail+str(i)+".png",np.reshape(pca.mean_,image_shape))
    
#pl.show()
