
# coding: utf-8

from __future__ import division
import os
import numpy as np
import nrrd
USERPATH = os.path.expanduser("~")
print(USERPATH)
import six.moves.cPickle as pickle
import random
import multiprocessing
num_cores = multiprocessing.cpu_count()

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline


# Load the dataset
f_Xdata = open('data_n2.save', 'rb')
f_Ydata = open('label_n2.save', 'rb')

X_data = pickle.load(f_Xdata)
print(X_data.shape)
X_data = X_data.astype('float32')

# normalize the raw data
X_data -= np.mean(X_data)
X_data /= np.std(X_data)
Y_data= pickle.load(f_Ydata)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y_data)
Y_data = encoder.transform(Y_data)

label = Y_data
data = X_data

print("Data shape and label shape")
print(data.shape, label.shape)

# init the global var
model = 0

def create_baseline():

    nb_classes = 1

    # create model
    global model
    model = Sequential()

    model.add(Convolution2D(10, 10, 2, border_mode='same',
                            input_shape=(1,50,40)))
    model.add(Activation('relu'))
    model.add(Convolution2D(10, 3, 3))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(40, 5, 3, border_mode='same' ))
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

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

seed = 7
# np.random.seed(seed)
estimators = []
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, nb_epoch=15,
                                          batch_size=64, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=label, n_folds=4, shuffle=True, random_state=seed)

results = cross_val_score(pipeline, data, label, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

json_string = model.to_json()
x = 2
model.save_weights('my_model_weights_2d_%d.h5'%x, overwrite=True)
open('my_model_architecture%d.json'%x, 'w').write(json_string)

