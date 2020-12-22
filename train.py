# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:47:26 2020

@author: Wei-Cheng Wong
"""

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import add, Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, AveragePooling2D, concatenate, Input, GlobalAveragePooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import cv2
from glob import glob
import pandas as pd
import random
import matplotlib.pyplot as plt
#%matplotlib inline

##########################################
# Parameters

# Image Size
img_height, img_width = 800, 400

batch_size = 5
CLASSNUM = 3

# Train, Validation, and Test Data Folders
train_data_dir = './train'
test_data_dir = './test'
val_data_dir = './val'

#######################################
# Preprocessing

random.seed(777)

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True
)

validation_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size
)

validation_generator = validation_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size
)

#########################################################################################################
# Model

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable = True):

    nb_filter1, nb_filter2 = filters
    bn_axis = 3
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (3, 3), name = conv_name_base + '2a', use_bias = True, padding = 'same', trainable = trainable)(input_tensor)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding = 'same', use_bias = True, name = conv_name_base + '2b', trainable = trainable)(x)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2b')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides = (2, 2), trainable = True):

    nb_filter1, nb_filter2 = filters
    bn_axis = 3
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (3, 3), strides = strides, use_bias = True, padding = 'same', name = conv_name_base + '2a', trainable = trainable)(input_tensor)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), use_bias = True, padding = 'same', name = conv_name_base + '2b', trainable = trainable)(x)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2b')(x)
    
    shortcut = Conv2D(nb_filter2, (1, 1), strides = strides, use_bias = True, name = conv_name_base + '1', trainable = trainable)(input_tensor)
    shortcut = BatchNormalization(axis = bn_axis, name = bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def Resnet18(width, height, depth, nb_classes, trainable = True):

    inpt = Input(shape = (height, width, depth))
    bn_axis = 3
    
    x = ZeroPadding2D((3, 3))(inpt)

    x = Conv2D(64, (7, 7), strides = (2, 2), use_bias = True, name = 'conv1', trainable = trainable)(x)
    x = BatchNormalization(axis = bn_axis, name = 'bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides = (2, 2))(x)

    x = conv_block(x, 3, [64, 64], stage = 2, block = 'a', strides = (1, 1), trainable = trainable)
    x = identity_block(x, 3, [64, 64], stage = 2, block = 'b', trainable = trainable)

    x = conv_block(x, 3, [128, 128], stage = 3, block = 'a', trainable = trainable)
    x = identity_block(x, 3, [128, 128], stage = 3, block = 'b', trainable = trainable)
    
    x = conv_block(x, 3, [256, 256], stage = 4, block = 'a', trainable = trainable)
    x = identity_block(x, 3, [256, 256], stage = 4, block = 'b', trainable = trainable)
    
    x = conv_block(x, 3, [512, 512], stage = 5, block = 'a', trainable = trainable)
    x = identity_block(x, 3, [512, 512], stage = 5, block = 'b', trainable = trainable)
    
    x = GlobalAveragePooling2D(name = 'avg_pool')(x)
    x = Dense(nb_classes, activation = 'softmax', name = 'fc1000')(x)
    
    model = Model(inpt, x)
    
    return model

####
model = Resnet18(img_width, img_height, 3, CLASSNUM)
print(model.summary())

####
opt = Adam(lr = 0.00001, decay = 1e-4)

nb_epochs = 20

model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

########
filepath = './check/epoch_weights.{epoch:02d}_loss.{val_loss:.2f}_acc.{val_accuracy:.2f}.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 5, verbose = 1)

History = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs,
    #class_weight = class_weight,
    callbacks = [checkpoint, reduce_lr, tbCallBack]
)
