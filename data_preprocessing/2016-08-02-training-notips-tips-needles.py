
# coding: utf-8

# In[4]:



# In[1]:

from __future__ import division
# import joblib
import glob
import os, re
import numpy as np
import nrrd
import numpy as np
from sklearn import datasets, svm, metrics, decomposition
from sklearn.externals import joblib
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

from keras import callbacks
remote = callbacks.RemoteMonitor(root='http://localhost:9000')


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
patchsizeSpaced = np.array(patchsize)//np.array(data_spacing)
patchsizeSpaced = [int(x) for x in patchsizeSpaced]
notipsPath = USERPATH + "/preprocessed_data/notips_%d-%d-%d_%.2f-%.2f-%.2f/" %(tuple(patchsize)+tuple(data_spacing))
tipsPath = USERPATH + "/preprocessed_data/tips_%d-%d-%d_%.2f-%.2f-%.2f/" %(tuple(patchsize)+tuple(data_spacing))
needlesPath = USERPATH + "/preprocessed_data/needles_%d-%d-%d_%.2f-%.2f-%.2f/" %(tuple(patchsize)+tuple(data_spacing))

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
    print("content: %d"%N)
    
    cubeTips = []
    data = []
    for path_i in cubeTipsPath:
        cubeTips.append(nrrd.read(path_i))
    for i in range(N):
        c = np.array(cubeTips[i][0][:,:,:])
        if c.shape==tuple(patchsizeSpaced):
            data.append(np.array(c))
    output = np.array(data, dtype='float32')
    print('number of sample %d' %len(output))
    return output


print(tipsPath)
# tips = loadAllDataFromPath(tipsPath, casesToExclude)
# notips = loadAllDataFromPath(notipsPath, casesToExclude)[:3*len(tips)]

o = 12


# In[5]:

tips = loadAllDataFromPath(tipsPath, casesToExclude)
notips = loadAllDataFromPath(notipsPath, casesToExclude)[:5*len(tips)]
needles = loadAllDataFromPath(needlesPath, casesToExclude)[:4*len(tips)]

print(len(tips), len(notips))

target_0 = [0 for i in range(len(notips))]
target_1 = [1 for i in range(len(tips))]
target_2 = [2 for i in range(len(needles))]
y_train = np.array(target_0 + target_1 + target_2)
print('target shape:', y_train.shape)
X_train = np.array(list(notips)+list(tips)+ list(needles))

print('data shape:', X_train.shape)

f_Xtrain = open('X_data_n%d.save'%o, 'wb')
f_ytrain = open('y_data_n%d.save'%o, 'wb')

pickle.dump(X_train, f_Xtrain, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(y_train, f_ytrain, protocol=pickle.HIGHEST_PROTOCOL)

f_Xtrain.close()
f_ytrain.close()


# In[6]:

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
print(y_data_binary)


# In[7]:

# In[7]:

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# init the global var
model = 0
m = 133
conv3d = False
conv1d = False
dimOrdering = 'tf'


# In[8]:

def create_baseline():

    nb_classes = 3

    # create model
    global model
        
    if m ==132:
        model = Sequential()
        model.add(Convolution2D(10, 10, 10, border_mode='same', input_shape=patchsizeSpaced, activation='relu', name='conv1_0'))
        model.add(ZeroPadding2D((2, 2)))
        model.add(Convolution2D(10, 5, 5, border_mode='same', activation='relu', name='conv1_1'))
        model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu', name='conv1_2'))
        model.add(MaxPooling2D((4,4), strides=(1,1)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='sigmoid'))
        
    if m ==133:
        model = Sequential()
        model.add(Convolution2D(10, 10, 10, border_mode='same', input_shape=patchsizeSpaced, activation='relu', name='conv1_0'))
#         model.add(ZeroPadding2D((2, 2)))
        model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu', name='conv1_1'))
        model.add(Convolution2D(10, 5, 5, border_mode='same', activation='relu', name='conv1_2'))
#         model.add(MaxPooling2D((4,4), strides=(1,1)))
        model.add(Dropout(0.2))
        
#         model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu', name='conv2_0'))
#         model.add(ZeroPadding2D((2, 2)))
#         model.add(Convolution2D(10, 5, 5, border_mode='same', activation='relu', name='conv2_1'))
#         model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu', name='conv2_2'))
#         model.add(MaxPooling2D((4,4), strides=(1,1)))
#         model.add(Dropout(0.2))
        
#         model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu', name='conv3_0'))
#         model.add(ZeroPadding2D((2, 2)))
#         model.add(Convolution2D(10, 5, 5, border_mode='same', activation='relu', name='conv3_1'))
#         model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu', name='conv3_2'))
#         model.add(MaxPooling2D((4,4), strides=(1,1)))
#         model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='sigmoid'))
        


    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

# np.random.seed(seed)
estimators = []
# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, nb_epoch=100,
                                          batch_size=512, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=y_data, n_folds=2, shuffle=True)#, random_state=seed)
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
results = cross_val_score(pipeline,X_data, y_data_binary, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

json_string = model.to_json()
model.save_weights('my_model_weights_2d_%d.h5'%m, overwrite=True)
open('my_model_architecture%d.json'%m, 'w').write(json_string)


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
# im = im[100//data_spacing[0]:160//data_spacing[0],80//data_spacing[1]:130//data_spacing[1],70//data_spacing[2]:160//data_spacing[2]]
im = im[100:160,80:130,70:160]
s = im.shape
print(s)
print(patchsizeSpaced)


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


# In[12]:

import pyprind
import sys
def gettips(N):
    '''
    Find the tip in the image by computing testing patches at every voxel position
    TODO: make this method more efficient
    '''
    p0, p1, p2 = patchsizeSpaced
    x0, y0, z0 = 0,0,0
    xe, ye, ze = s
    
    tips = []
    bar = pyprind.ProgBar(s[0], title='Find_tip', stream=sys.stdout)
    res = []
    for xi in range(x0, xe-p0):
        for yi in range(y0, ye-p1):
            vols = [im[xi:xi+p0,yi:yi+p1,zi:zi+p2] for zi in range(z0,ze-p2)]
            # we normalize the data (centered on mean 0 and rescaled in function of the STD)
            volnorm = vols - np.mean(vols)
            volnorm2 = volnorm/np.std(volnorm)
            cube = np.array(volnorm2)
            if not conv1d and dimOrdering == 'th':
                cube = np.swapaxes(cube, 1,3)
            if conv3d:
                cube = np.expand_dims(cube,1)
            if conv1d:
                cube = cube.reshape(cube.shape[0], cube.shape[1]*cube.shape[2],cube.shape[3])
                
            res.append(model.predict_proba(cube, batch_size=ze-p2-z0, verbose=False))
        bar.update()
    return np.array(res)

def findtips(res, prob):
    N=1
    p0, p1, p2 = patchsizeSpaced
    
    x0, y0, z0 = 0,0,0
    xe, ye, ze = s
    
    i = -1
    tips = []
    for xi in range(x0, xe-p0):
        for yi in range(y0, ye-p1):
            i+=1
            indices = np.where(res[i][:,0]>=prob)
            # we add the coordinates of the center voxel of the patches that tested positive
            for z in indices[0]:
                tips.append([xi+p0/2, yi+p1/2, z0+p2/2+z])
    return tips

def findtips3classes(res, prob, cat):
    N=1
    res = res[:,:,cat]
    p0, p1, p2 = patchsizeSpaced
    
    x0, y0, z0 = 0,0,0
    xe, ye, ze = s
    
    i = -1
    tips = []
    for xi in range(x0, xe-p0-1):
        for yi in range(y0, ye-p1-1):
            i+=1
            indices = np.where(res[i][:]>=prob)
            # we add the coordinates of the center voxel of the patches that tested positive
            for z in indices[0]:
                tips.append([xi+p0/2, yi+p1/2, z0+p2/2+z])

    return tips

def findtips3classes2(res, prob):
    N=1
    res = (res[:,:,2] + res[:,:,1])/2
    p0, p1, p2 = patchsizeSpaced

    x0, y0, z0 = 0,0,0
    xe, ye, ze = s
    
    i = -1
    tips = []
    for xi in range(x0, xe-p0):
        for yi in range(y0, ye-p1):
            i+=1
            indices = np.where(res[i][:]>=prob)
            # we add the coordinates of the center voxel of the patches that tested positive
            for z in indices[0]:
                tips.append([xi+p0/2, yi+p1/2, z0+p2/2+z])
    return tips


# In[13]:

# find the tips for patches with size p
print(patchsizeSpaced)
pred=gettips(1)


# In[14]:


np.set_printoptions(precision=2)
pred [ pred < 0.1] = 0
print(pred.shape)
print(len(pred[:,:,1]))
print(pred[:,:,1])
# res = findtips(pred, 1)
res = findtips3classes(pred, 1,2)


# In[15]:

res[30]


# ## Creation of a labelmap from the voxel that tested positive

# In[16]:

mask = np.zeros(im.shape)
for coord in res:
    mask[int(coord[0]),int(coord[1]),int(coord[2])]=1.0
nrrd.write('mask%d.nrrd'%m, mask)
nrrd.write('im%d.nrrd'%m, im)


# In[17]:

get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')
# We display one axial slice
Z = 18
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(121)
ax1.imshow((np.clip(mask[:,:,Z]*255+im[:,:,Z]/2,a_min=0,a_max=200)).transpose(),  cmap='gray', interpolation='nearest')
ax2 = fig.add_subplot(122)
ax2.imshow(im[:,:,Z].transpose(), cmap='gray', interpolation='nearest')


# In[13]:

#     if m ==7:
#         model = Sequential()

#         model.add(Convolution2D(10, 10, 2, border_mode='same',
#                                 input_shape=(10,10,10)))
#         model.add(Activation('relu'))
#         model.add(Convolution2D(10, 3, 3))
#         model.add(Activation('relu'))
#         # model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.1))

#         model.add(Convolution2D(40, 5, 3, border_mode='same' ))
#         model.add(Activation('relu'))
#         model.add(Convolution2D(40, 5, 3, border_mode='same'))
#         model.add(Activation('relu'))
#         model.add(Convolution2D(40, 5, 3))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.1))

#         model.add(Flatten())
#         model.add(Dense(480))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(480))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(nb_classes))
#         model.add(Activation('sigmoid'))

#     if m == 11:
#         model = Sequential()
#         model.add(Convolution2D(10, 10, 10, border_mode='same',
#                                 batch_input_shape=(10, 10, 10)))
#         # model.add(ZeroPadding2D((1, 1), batch_input_shape=(1, 3, 10, 10)))
#         model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
#         model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
#         model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
#         model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
#         model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
#         model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#         model.add(Flatten(input_shape=(512,3,3)))
#         model.add(Dense(480))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(480))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(nb_classes))
#         model.add(Activation('sigmoid'))

#     if m ==13:
#         model = Sequential()
#         model.add(Convolution2D(10, 10, 10, border_mode='same', input_shape=(10,10,10), activation='relu', name='conv1_0'))
# #         model.add(ZeroPadding2D((2, 2)))
#         model.add(Convolution2D(10, 5, 5, border_mode='same', activation='relu', name='conv1_1'))
#         model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu', name='conv1_2'))
#         model.add(MaxPooling2D((4,4), strides=(1,1)))
#         model.add(Dropout(0.5))

#         model.add(Flatten())
#         model.add(Dense(200, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(100, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(nb_classes, activation='sigmoid'))
        
#     if m ==132:
#         model = Sequential()
#         model.add(Convolution2D(10, 10, 10, border_mode='same', input_shape=(10,10,10), activation='relu', name='conv1_0'))
# #         model.add(ZeroPadding2D((2, 2)))
#         model.add(Convolution2D(10, 5, 5, border_mode='same', activation='relu', name='conv1_1'))
#         model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu', name='conv1_2'))
#         model.add(MaxPooling2D((4,4), strides=(1,1)))
#         model.add(Dropout(0.5))

#         model.add(Flatten())
#         model.add(Dense(200, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(100, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(nb_classes, activation='sigmoid'))
        
#     if m ==132:
#         model = Sequential()
#         model.add(Convolution2D(10, 10, 10, border_mode='same', input_shape=(10,10,10), activation='relu', name='conv1_0'))
# #         model.add(ZeroPadding2D((2, 2)))
#         model.add(Convolution2D(10, 5, 5, border_mode='same', activation='relu', name='conv1_1'))
#         model.add(Convolution2D(10, 10, 10, border_mode='same', activation='relu', name='conv1_2'))
#         model.add(MaxPooling2D((4,4), strides=(1,1)))
#         model.add(Dropout(0.5))

#         model.add(Flatten())
#         model.add(Dense(200, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(100, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(nb_classes, activation='sigmoid'))
        
#     if m ==14:
#         model = Sequential()
#         model.add(Convolution2D(20, 10, 10, border_mode='same', input_shape=(10,10,20), activation='relu', name='conv1_0'))
#         model.add(Convolution2D(20, 5, 5, border_mode='same', activation='relu', name='conv1_1'))
#         model.add(Convolution2D(20, 10, 10, border_mode='same', activation='relu', name='conv1_2'))
#         model.add(MaxPooling2D((4,4), strides=(1,1)))
#         model.add(Dropout(0.5))
        
#         model.add(Convolution2D(20, 10, 10, border_mode='same', input_shape=(10,10,20), activation='relu', name='conv2_0'))
#         model.add(Convolution2D(20, 5, 5, border_mode='same', activation='relu', name='conv2_1'))
#         model.add(Convolution2D(20, 10, 10, border_mode='same', activation='relu', name='conv2_2'))
#         model.add(MaxPooling2D((4,4), strides=(1,1)))
#         model.add(Dropout(0.5))

#         model.add(Flatten())
#         model.add(Dense(200, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(100, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(nb_classes, activation='sigmoid'))
        
        
#     if m ==15:
#         model = Sequential()

#         model.add(Convolution2D(100,2,2, border_mode='same', 
#                                 input_shape=(20,10,10)))
#         model.add(Activation('relu'))
#         model.add(Convolution2D(50,2,2))
# #         model.add(Activation('relu'))
# #         model.add(Convolution2D(60,4,4))
# #         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2,2)))
#         model.add(Dropout(0.25))

#         model.add(Flatten())

#         model.add(Dense(480))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(nb_classes))
#         model.add(Activation('sigmoid'))
    
#     if m == 16:
#         model = Sequential()

#         model.add(Convolution3D(20,9,9,18, border_mode='same',
#                                 input_shape=(1,10,10,20)))
#         model.add(Activation('relu'))
#         model.add(Convolution3D(20,9,9,19))
#         model.add(Activation('relu'))
#         model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# #         model.add(Dropout(0.25))

#         model.add(Flatten())
#         model.add(Dense(480))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(nb_classes))
#         model.add(Activation('sigmoid'))
        
#     if m ==18:
#         model = Sequential()

#         model.add(ZeroPadding2D((1,1),input_shape=(20,10,10)))
#         model.add(Convolution2D(64, 3, 3, activation='relu'))
#         model.add(ZeroPadding2D((1,1)))
#         model.add(Convolution2D(64, 3, 3, activation='relu'))
#         model.add(MaxPooling2D((2,2), strides=(2,2)))
        
#         model.add(ZeroPadding2D((1,1)))
#         model.add(Convolution2D(128, 3, 3, activation='relu'))
#         model.add(ZeroPadding2D((1,1)))
#         model.add(Convolution2D(128, 3, 3, activation='relu'))
#         model.add(MaxPooling2D((2,2), strides=(2,2)))

#         model.add(ZeroPadding2D((1,1)))
#         model.add(Convolution2D(256, 3, 3, activation='relu'))
#         model.add(ZeroPadding2D((1,1)))
#         model.add(Convolution2D(256, 3, 3, activation='relu'))
#         model.add(ZeroPadding2D((1,1)))
#         model.add(Convolution2D(256, 3, 3, activation='relu'))
#         model.add(MaxPooling2D((2,2), strides=(2,2)))

#         model.add(ZeroPadding2D((1,1)))
#         model.add(Convolution2D(512, 2, 2, activation='relu'))
#         model.add(ZeroPadding2D((1,1)))
#         model.add(Convolution2D(512, 2, 2, activation='relu'))
#         model.add(ZeroPadding2D((1,1)))
#         model.add(Convolution2D(512, 2, 2, activation='relu'))
#         model.add(MaxPooling2D((2,2), strides=(2,2)))

#         model.add(Flatten())
#         model.add(Dense(4096, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(4096, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(1, activation='sigmoid'))
        
        
#     if m ==20:
#         model = Sequential()
#         model.add(Convolution1D(100,10, activation='relu', input_shape=(100,10)))
#         model.add(Dropout(0.5))
#         model.add(Convolution1D(100,10, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Convolution1D(100,10, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Convolution1D(100,10, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(MaxPooling1D((5)))


# In[ ]:



