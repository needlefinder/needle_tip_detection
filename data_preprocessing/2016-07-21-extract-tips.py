
# coding: utf-8

# # Extract needle tips and random patchs

# In[1]:

from __future__ import division
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'pylab inline')

import pyprind
import glob
import os
import SimpleITK as sitk
import numpy as np
import nrrd
import numpy as np
from sklearn import datasets, svm, metrics, decomposition
from sklearn.externals import joblib
import time
from joblib import Parallel, delayed  
import multiprocessing
num_cores = multiprocessing.cpu_count()
USERPATH = os.path.expanduser("~")
print(USERPATH)
from mpl_toolkits.mplot3d import Axes3D
import six.moves.cPickle as pickle


# 

# In[155]:

def getPaths(spacing):
    labelPath = USERPATH + '/Projects/LabelMaps_%.2f-%.2f-%.2f/*' % tuple(spacing)
    print(labelPath)
    casePath = glob.glob(labelPath)
    volumePath = [cp + '/case.nrrd' for cp in casePath]
    return volumePath, casePath

def getVolumeAndNeedle(k):
    '''
    Outputs the volume and needles paths
    '''
    needlePath  = glob.glob(casePath[k] + '/needle*.nrrd')
    return volumePath[k], needlePath

def getTipsPos(ndls):
    '''
    Read the needle labelmaps and outputs the tip positions
    '''
    needleTips = []
    for n in ndls: 
        ndl = nrrd.read(n)[0]
        ztest = (np.where(ndl>0)[2])
        if len(ztest):
            zmax = np.where(ndl>0)[2].max()
            xl,yl = np.where(ndl[...,zmax]>0)
            xmax = int(np.mean(xl))
            ymax = int(np.mean(yl))
            needleTips.append([xmax, ymax, zmax])
    return needleTips

def getTipsPath(casePath, spacing):
    '''
    Returns path where to save the tips
    '''
    return USERPATH + '/Projects/tips_%d-%d-%d_%.2f-%.2f-%.2f/%s' %(tuple(patchsize)+tuple(spacing)+(casePath.split('/')[-1],))

def getNoTipsPath(casePath, spacing):
    '''
    Return path where to save the random patchs
    '''
    return USERPATH + '/Projects/notips_%d-%d-%d_%.2f-%.2f-%.2f/%s' %(tuple(patchsize)+tuple(spacing)+(casePath.split('/')[-1],))

def saveTips(tipsPos, casePath, spacing, patchsize):
    '''
    Save the tips
    '''
    tipPath = getTipsPath(casePath, spacing)
    vol = nrrd.read(casePath + '/case.nrrd')[0]
    for i, tipPos in enumerate(tipsPos):
        x, y, z = tipPos
        xmin, ymin, zmin = np.array(patchsize)//2
        xmin = xmin//spacing[0]
        ymin = ymin//spacing[1]
        zmin = zmin//spacing[2]
        tip = vol[x-xmin:x+xmin, y-ymin:y+ymin, z-zmin:z+zmin]
        createDir(tipPath)
        nrrd.write(tipPath + '/tip-%d.nrrd'%i, tip)
        
def saveNoTips(numberOfSamples, casePath, spacing, patchsize):
    '''
    Save the random cubes
    '''
    vol = nrrd.read(casePath + '/case.nrrd')[0]
    tipPath = getNoTipsPath(casePath, spacing)
    for i in range(numberOfSamples):
        xmin, ymin, zmin = np.array(patchsize)//2
        xmin = xmin//spacing[0]
        ymin = ymin//spacing[1]
        zmin = zmin//spacing[2]
        x = np.random.randint(xmin,vol.shape[0]-xmin)
        y = np.random.randint(ymin,vol.shape[1]-ymin)
        z = np.random.randint(zmin,vol.shape[2]-zmin)
        
        tip = vol[x-xmin:x+xmin, y-ymin:y+ymin, z-zmin:z+zmin]
        createDir(tipPath)
        nrrd.write(tipPath + '/notip-%d.nrrd'%i, tip)
        
def createDir(directory):
    '''
    Create a directory if it doesn't exist. Do nothing otherwise.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
    return 0

def extractTips(spacing, patchsize):
    '''
    Extract all the tips from all the labelmaps
    '''
    _, casePath = getPaths(spacing)
    for k, path in enumerate(casePath):
        _, ndls = getVolumeAndNeedle(k)
        tipsPos = getTipsPos(ndls)
        saveTips(tipsPos, casePath[k], spacing, patchsize)
        
def extractNoTips(spacing, patchsize, numberOfSamples):
    '''
    Extract random cases
    '''
    _, casePath = getPaths(spacing)
    for k, path in enumerate(casePath):
        _, ndls = getVolumeAndNeedle(k)
        saveNoTips(numberOfSamples, casePath[k], spacing, patchsize)


# In[ ]:

for spacing in [[1,1,1], [0.6,0.6,1]]:
    for patchsize in [[10,10,10], [10,10,20], [5,5,20], [20,20,20]]:
        extractTips(spacing, patchsize)
        extractNoTips(spacing, patchsize, 200)

