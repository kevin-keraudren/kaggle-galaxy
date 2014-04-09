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

import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt


def to_dict(responses,id_responses):
    res = {}
    for r,i in zip(responses,id_responses):
        res[i] = r
    return res

f = "/vol/biomedic/users/kpk09/kaggle/galaxy/data/training_solutions_rev1.csv"
def get_mapping( f ):
    reader = csv.DictReader( open( f, 'rb' ) )
    for line in reader:
        mapping = sorted(line.keys())
        break
    return mapping

mapping = get_mapping(f)
mapping = [mapping[-1]] + mapping[:-1]

responses,id_responses = galaxy.read_responses(f)

step = 4
for i in range(37):
    for j in range(i+1,37):
        fig, ax = plt.subplots()
        ax.scatter(responses[::step,i], responses[::step,j], alpha=0.1)

        plt.xlim(0.0,1.0)
        plt.ylim(0.0,1.0)
        ax.set_xlabel(mapping[i+1], fontsize=20)
        ax.set_ylabel(mapping[j+1], fontsize=20)

        ax.grid(True)
        fig.tight_layout()

        plt.savefig(mapping[i+1] + "-" + mapping[j+1] + ".png")
        plt.close(fig)
