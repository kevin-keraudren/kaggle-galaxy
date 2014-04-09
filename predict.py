#!/usr/bin/python

import sys, os
import cv2
import numpy as np
import scipy.ndimage as nd
import csv

import joblib
from joblib import Parallel, delayed

import galaxy
from glob import glob

what = "testing"
if len(sys.argv) > 1:
    what = sys.argv[1]

if what == "training": 
    folder = "/vol/biomedic/users/kpk09/kaggle/galaxy/data/images_training_rev1/"
else:
    folder = "/vol/biomedic/users/kpk09/kaggle/galaxy/data/images_test_rev1/"

f = "/vol/biomedic/users/kpk09/kaggle/galaxy/data/training_solutions_rev1.csv"

files = glob(folder+"/*")

print "will predict", len(files), "galaxies" 
print "from", folder

points = Parallel(n_jobs=-1)(delayed(galaxy.get_features)(f)
                             for f in files )

mapping = galaxy.get_fieldnames()

forest = joblib.load(galaxy.get_data_folder()+"/galaxy_forest")
forest.set_params(n_jobs=1)

scaler2 = joblib.load(galaxy.get_data_folder()+"/scaler2")
points = scaler2.transform(points)

predictions = forest.predict( points )

scaler1 = joblib.load(galaxy.get_data_folder()+"/scaler1")
predictions = scaler1.inverse_transform(predictions)

# sparsify
# min on training responses: 2.9099999e-06
predictions[predictions<1e-7] = 0

#f = "/vol/biomedic/users/kpk09/kaggle/galaxy/data/training_solutions_rev1.csv"
#responses, ids = galaxy.read_responses( f )

# standardization
#mean = responses.mean(axis=0)
#responses -= mean

#predictions /= 1000
#predictions *= responses.std(axis=0)
#predictions += mean

all_res = []
for f, prediction in zip(files,predictions):
    res = {}
    res['GalaxyID'] = os.path.basename(f)[:-len(".jpg")]
    for i in xrange(37):
        res[mapping[i+1]] = prediction[i]
    all_res.append(res)

writer = csv.DictWriter(open(galaxy.get_data_folder()+"/predictionsSVM.csv","w"),
                        fieldnames=mapping)
writer.writeheader()

for res in all_res:
    writer.writerow(res)
