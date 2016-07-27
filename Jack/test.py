
# coding: utf-8

# # Find TIPS
# ## After training the network, load the weights and apply network to classify each voxel of a test cases


from __future__ import division

import sys
import pyprind
import os
import nrrd
import numpy as np
import random
import multiprocessing
num_cores = multiprocessing.cpu_count()
USERPATH = os.path.expanduser("~")
print(USERPATH)
import six.moves.cPickle as pickle
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
# ## Load pre-processed data to check network performance

# dimension of the patch
p1 = 10 
p2 = 10
p3 = 20
# weight number
x = 2

f_Xdata = open('data_n3.save', 'rb')
f_Ydata = open('label_n3.save', 'rb')

# we load the data via pickle
X_data = pickle.load(f_Xdata)
Y_data= pickle.load(f_Ydata)

# we shuffle the data
index = [i for i in range(len(X_data))]
random.shuffle(index)
X_data = X_data[index]
Y_data = Y_data[index]


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y_data)
Y_data = encoder.transform(Y_data)

# we make sure the data is in the right format
X_data = X_data.astype('float32')
X_data  -= np.mean(X_data)
X_data /= np.std(X_data)

# data = [[] for i in range(len(X_data))]
# for i in range(len(X_data)):
#     temp = X_data[i].reshape(50,40)
#     data[i] = [temp]
# X_data = np.array(data)

X_test = X_data
Y_test = Y_data

# we load the model with the trained weights
model = model_from_json(open('my_model_architecture%d.json'%x).read())
model.load_weights('my_model_weights_2d_%d.h5'%x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# we check what was the performance of this CNN on the trained data
Y_predict = model.predict_classes(X_test[:][:], batch_size=64)
print('predict error:')
print('1 for tip, 0 for notip:')
tip = 0
notip = 0
for i in range(len(Y_predict)):
    if Y_test[i] != Y_predict[i]:
        if Y_test[i] == 1:
            tip += 1
        else:
            notip += 1
        print(Y_predict[i], Y_test[i])
print('there are ', len(Y_predict), ' samples!')
print('error: ', tip, 'tips and ', notip, 'notips')

# we load a test case
nrrdData = nrrd.read('Case033.nrrd')
im = nrrdData[0]
# choose the reasonable region of test patch
im = im[301:375, 207:310, 84:]
s = im.shape
print(s)

def findtips(N, p1, p2, p3):
    '''
    Find the tip in the image by computing testing patches at every voxel position
    TODO: make this method more efficient
    '''
    xmiddle = s[0]//2
    ymiddle = s[1]//2
    zmiddle = s[2]//2
    
    x0= xmiddle - xmiddle//N
    y0= ymiddle - ymiddle//N
    z0= zmiddle - zmiddle//N
    
    xe= xmiddle + xmiddle//N
    ye= ymiddle + ymiddle//N
    ze= zmiddle + zmiddle//N
    
    tips = []
    bar = pyprind.ProgBar(xmiddle//N*2, title='Find_tip', stream=sys.stdout)
    for xi in range(x0, xe-p1):
        for yi in range(y0, ye-p2):
            vols = [im[xi:xi+p1,yi:yi+p2,zi:zi+p3] for zi in range(z0,ze-p3)]
            # we normalize the data (centered on mean 0 and rescaled in function of the STD)
            volnorm = [ x-np.mean(x) for x in vols]
            volnorm2 = [x/np.std(x) for x in volnorm]
            vol = np.array(volnorm2)
            cube = [[] for i in range(len(vol))]
            for i in range(len(vol)):
            	temp = vol[i].reshape(50,40)
            	cube[i] = [temp]
            cube = np.array(cube)
            res = model.predict_proba(cube, batch_size=32, verbose=False)
            indices = np.where(res[:,0]==1)
            # we add the coordinates of the center voxel of the patches that tested positive
            if len(indices[0]) >= 1:
                for i in range(1):
                    tips.append([xi+p1/2,yi+p2/2,z0+p3/2+indices[0][-i-1]])
        bar.update()
    return tips



# find the tips for patches with size p
res = findtips(1, p1, p2, p3)
res = np.array(res)
print(res.shape)


# ## Creation of a labelmap from the voxel that tested positive
mask = np.zeros(im.shape)
for coord in res:
    mask[int(coord[0]),int(coord[1]),int(coord[2])]=1.0
nrrd.write('mask%d-99.nrrd'%x, mask)
nrrd.write('im%d-99.nrrd'%x, im)
