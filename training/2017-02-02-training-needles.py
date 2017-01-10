
# coding: utf-8

# In[1]:



# In[1]:

from __future__ import division
# import joblib
import glob
import os, re
import numpy as np
import nrrd
import numpy as np
from sklearn import datasets, svm, metrics, decomposition
# from sklearn.externals import joblib
import time
# from joblib import Parallel, delayed  
import multiprocessing
num_cores = multiprocessing.cpu_count()
USERPATH = os.path.expanduser("~")
print(USERPATH)
import six.moves.cPickle as pickle
# import tensorflow as tf

# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding1D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mpl_toolkits.mplot3d import Axes3D



# server = tf.train.Server.create_local_server()
# sess = tf.Session(server.target)

# from keras import backend as K
# K.set_session(sess)

# tb = TensorBoard(log_dir='/tmp/tensorboard', histogram_freq=1, write_graph=True)



# fix random seed for reproducibility
seed = 7
# checkpointer = ModelCheckpoint(filepath="weights2d.hdf5", verbose=1, save_best_only=True)
patchsize = [10,10,10]
data_spacing = [1,1,1]
notipsPath = USERPATH + "/preprocessed_data/notips_%d-%d-%d_%.2f-%.2f-%.2f/" %(tuple(patchsize)+tuple(data_spacing))
tipsPath = USERPATH + "/preprocessed_data/needles_%d-%d-%d_%.2f-%.2f-%.2f/" %(tuple(patchsize)+tuple(data_spacing))

casesToExclude = [64,77]


def getTrainingPaths(tipsPath, cases=[64,77]):
    strL = ""
    for c in cases:
        strL+="%03d|"%c
    fnames=glob.glob(tipsPath + "/*/*.nrrd")
    regex=re.compile("^((?!%s).)*$"%strL[:-1])
    paths = [m.group(0) for l in fnames for m in [regex.search(l)] if m]
    return paths

def loadAllDataFromPath(path, casesToExclude):
    # path in directorty
    
#     cubeTipsPath = glob.glob(path + "/*/*.nrrd")
    cubeTipsPath = getTrainingPaths(path, casesToExclude)
    # number of samples
    N = len(cubeTipsPath)
    
    cubeTips = []
    data = []
    for path_i in cubeTipsPath:
        cubeTips.append(nrrd.read(path_i))
    for i in range(N):
        # c = np.array(cubeTips[i][0])  # for patches of size 20,20,20
        c = np.array(cubeTips[i][0][:,:,:]) # for patches of size 10,10,10
        if c.shape==tuple(patchsize):
            data.append(np.array(c))
    output = np.array(data, dtype='float32')
    print('number of sample %d' %len(output))
    return output


print(tipsPath)
# tips = loadAllDataFromPath(tipsPath, casesToExclude)
# notips = loadAllDataFromPath(notipsPath, casesToExclude)[:3*len(tips)]


# In[2]:

tips = loadAllDataFromPath(tipsPath, casesToExclude)
notips = loadAllDataFromPath(notipsPath, casesToExclude)[:5*len(tips)]

print(len(tips), len(notips))

target_0 = [0 for i in range(len(notips))]
target_1 = [1 for i in range(len(tips))]
y_train = np.array(target_0 + target_1)
print('target shape:', y_train.shape)
X_train = np.array(list(notips)+list(tips))

print('data shape:', X_train.shape)


# In[6]:

o = 10
f_Xtrain = open('X_data_n%d.save'%o, 'wb')
f_ytrain = open('y_data_n%d.save'%o, 'wb')

pickle.dump(X_train, f_Xtrain, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(y_train, f_ytrain, protocol=pickle.HIGHEST_PROTOCOL)

f_Xtrain.close()
f_ytrain.close()


# In[6]:

# Load the dataset
f_Xdata = open('X_data_n%d.save'%o, 'rb')
f_ydata = open('y_data_n%d.save'%o, 'rb')

X_data_ = pickle.load(f_Xdata)
X_data_ = X_data_.astype('float32')

# normalize the raw data
X_data_ -= np.mean(X_data_)
X_data_ /= np.std(X_data_)

## second method for normalization
# X_data /= 255

y_data= pickle.load(f_ydata)
y_data_binary = to_categorical(y_data)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_data)
y_data = encoder.transform(y_data)

print("Data shape and label shape")
print(X_data_.shape, y_data.shape)


# In[7]:

# In[7]:

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# init the global var
model = 0
m = 13
conv3d = False
conv1d = False
dimOrdering = 'tf'


# In[8]:

import sys
oldstdout = sys.stdout
sys.stdout = sys.__stdout__
def create_baseline():

    nb_classes = 1

    # create model
    global model
    if m ==7:
        model = Sequential()

        model.add(Convolution2D(10, 10, 2, border_mode='same',
                                input_shape=(10,10,10)))
        model.add(Activation('relu'))
        model.add(Convolution2D(10, 3, 3))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Convolution2D(40, 5, 3, border_mode='same' ))
        model.add(Activation('relu'))
        model.add(Convolution2D(40, 5, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(40, 5, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Flatten())
        model.add(Dense(480))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(480))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('sigmoid'))

    if m ==13:
        model = Sequential()
        model.add(Convolution2D(10, 10, 10, border_mode='same', input_shape=(10,10,10), activation='relu', name='conv1_0'))
        model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu', name='conv1_1'))
        model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu', name='conv1_2'))
        model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu', name='conv1_3'))
        model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu'))
#         model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu'))
#         model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu'))
#         model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='sigmoid'))
        

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

# np.random.seed(seed)
estimators = []
# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, nb_epoch=30,
                                          batch_size=2000, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=y_data, n_folds=3, shuffle=True)#, random_state=seed)
if not conv1d and dimOrdering == 'th':
    X_data = np.swapaxes(X_data_,1,3)
    X_data = np.swapaxes(X_data,2,3)
    print(X_data.shape)
elif conv1d:
    print(X_data_.shape)
    X_data = X_data_.reshape((X_data_.shape[0], X_data_.shape[1]* X_data_.shape[2], X_data_.shape[3]))
    print(X_data.shape)
else:
    X_data = X_data_

if conv3d:
    X_data =  np.expand_dims(X_data, 1)
    
print(X_data.shape)
results = cross_val_score(pipeline,X_data, y_data, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

json_string = model.to_json()
model.save_weights('my_model_weights_2d_%d_gp_2.h5'%m, overwrite=True)
open('my_model_architecture%d_gp_2.json'%m, 'w').write(json_string)


# In[9]:

# we load a test case and the model

# model = model_from_json(open('my_model_architecture%d.json'%m).read())
# model.load_weights('my_model_weights_2d_%d.h5'%m)
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
print(data_spacing)
nrrdData = nrrd.read(USERPATH + '/preprocessed_data/LabelMaps_%.2f-%.2f-%.2f/064/case.nrrd'%(tuple(data_spacing)))
im = nrrdData[0]
im = im[100:160,80:130,70:160]
s = im.shape
print(s)
p=10
print(patchsize)


# In[10]:

# import pyprind
# import sys
# def findtips(N):
#     '''
#     Find the tip in the image by computing testing patches at every voxel position
#     TODO: make this method more efficient
#     '''
#     p0, p1, p2 = patchsize
#     xmiddle = s[0]//2
#     ymiddle = s[1]//2
#     zmiddle = s[2]//2
    
#     x0= xmiddle - xmiddle//N
#     y0= ymiddle - ymiddle//N
#     z0= zmiddle - zmiddle//N
    
#     xe= xmiddle + xmiddle//N
#     ye= ymiddle + ymiddle//N
#     ze= zmiddle + zmiddle//N
    
#     tips = []
#     bar = pyprind.ProgBar(xmiddle//N*2, title='Find_tip', stream=sys.stdout)
#     for xi in range(x0, xe-p0):
#         for yi in range(y0, ye-p1):
#             vols = [im[xi:xi+p0,yi:yi+p1,zi:zi+p2] for zi in range(z0,ze-p2)]
#             # we normalize the data (centered on mean 0 and rescaled in function of the STD)
#             volnorm = [ x-np.mean(x) for x in vols]
#             volnorm2 = [x/np.std(x) for x in volnorm]
#             cube = np.array(volnorm2)
#             cube = np.swapaxes(cube, 1,3)
# #             cube = np.swapaxes(cube, 2,3)
#             if conv3d:
#                 cube = np.expand_dims(cube,1)
#             res = model.predict_proba(cube, batch_size=ze-p2-z0, verbose=False)
#             indices = np.where(res[:,0]==1)
#             # we add the coordinates of the center voxel of the patches that tested positive
#             for z in indices[0]:
#                 tips.append([xi+p0/2,yi+p1/2,z0+p2/2+z])
#         bar.update()
#     return tips


# In[11]:

import pyprind
import sys
def gettips(N):
    '''
    Find the tip in the image by computing testing patches at every voxel position
    TODO: make this method more efficient
    '''
    p0, p1, p2 = patchsize
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
    res = []
    for xi in range(x0, xe-p0):
        for yi in range(y0, ye-p1):
            vols = [im[xi:xi+p0,yi:yi+p1,zi:zi+p2] for zi in range(z0,ze-p2)]
            # we normalize the data (centered on mean 0 and rescaled in function of the STD)
            volnorm = vols - np.mean(vols)
            volnorm2 = volnorm/np.std(volnorm)
#             volnorm = [ x-np.mean(x) for x in vols]
#             volnorm2 = [x/np.std(x) for x in volnorm]
            cube = np.array(volnorm2)
            if not conv1d and dimOrdering == 'th':
                cube = np.swapaxes(cube, 1,3)
            if conv3d:
                cube = np.expand_dims(cube,1)
            if conv1d:
                cube = cube.reshape(cube.shape[0], cube.shape[1]*cube.shape[2],cube.shape[3])
            res.append(model.predict_proba(cube, batch_size=ze-p2-z0, verbose=False))
        bar.update()
    return res

def findtips(res, prob):
    N=1
    p0, p1, p2 = patchsize
    xmiddle = s[0]//2
    ymiddle = s[1]//2
    zmiddle = s[2]//2
    
    x0= xmiddle - xmiddle//N
    y0= ymiddle - ymiddle//N
    z0= zmiddle - zmiddle//N
    
    xe= xmiddle + xmiddle//N
    ye= ymiddle + ymiddle//N
    ze= zmiddle + zmiddle//N
    
    i = -1
    tips = []
    for xi in range(x0, xe-p0):
        for yi in range(y0, ye-p1):
            i+=1
            indices = np.where(res[i][:,0]>=prob)
            # we add the coordinates of the center voxel of the patches that tested positive
            for z in indices[0]:
                tips.append([xi+p0/2,yi+p1/2,z0+p2/2+z])
    return tips


# In[12]:

# find the tips for patches with size p
pred=gettips(1)


# In[13]:

print(len(pred))
res = findtips(pred, 0.99999)
len(res)


# ## Creation of a labelmap from the voxel that tested positive

# In[14]:

mask = np.zeros(im.shape)
for coord in res:
    mask[int(coord[0]),int(coord[1]),int(coord[2])]=1.0
nrrd.write('mask%d.nrrd'%m, mask)
nrrd.write('im%d.nrrd'%m, im)


# In[16]:

import matplotlib.pylab as plt
get_ipython().magic('matplotlib notebook')

# %pylab inline
# We display one axial slice
Z = 30
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(121)
ax1.imshow((np.clip(mask[:,:,Z]*255+im[:,:,Z]/2,a_min=0,a_max=200)).transpose(),  cmap='gray', interpolation='nearest')
ax2 = fig.add_subplot(122)
ax2.imshow(im[:,:,Z].transpose(), cmap='gray', interpolation='nearest')


# In[17]:

xs,ys,zs = np.where(mask==1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, marker='o', alpha=0.3, s=10)


# In[ ]:



