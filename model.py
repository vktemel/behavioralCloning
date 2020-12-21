# Import used libraries
import pandas as pd
import numpy as np
import cv2
import os
import csv
import matplotlib.pyplot as plt
import sklearn
from math import ceil
from sklearn.model_selection import train_test_split
import random

# This setting is applied to remove "CUDNN_STATUS_ALLOC_FAILED" error on my local environment.
# For more info: https://stackoverflow.com/a/65203824
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Generator to yield X, y dataset
# This generator is created to reduce the memory consumption while training the model
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # Shuffle the samples
        sklearn.utils.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            # Create empty lists
            images = []
            measurements = []

            # For each sample in the batch
            for batch_sample in range(len(batch_samples)):
                # Get the full path of the image
                imgName = batch_samples.iloc[batch_sample].img_path
                imgPath = img_folder + imgName
                # Store the image and steering
                img = cv2.imread(imgPath)
                meas = batch_samples.iloc[batch_sample].steering
                # Append the original image
                images.append(img)
                measurements.append(meas)
                # Augment data by flipping the images and invert measurements
                images.append(cv2.flip(img,1))
                measurements.append(meas*-1.0)

            # Convert to numpy arrays
            X = np.array(images)
            y = np.array(measurements)
            # Shuffle again and yield
            X, y = sklearn.utils.shuffle(X, y)
            yield X, y

# Set the folder paths
data_folder = 'data\\'
img_folder = data_folder + 'IMG\\'
log_path = data_folder + 'driving_log.csv'

# Read the csv file as a pandas dataframe, and set the name of the columns
column_names = ['Center_Image', 'Left_Image', 'Right_Image', 'Steering', 'Throttle', 'Brake', 'Speed']
df_log = pd.read_csv(log_path, names=column_names)

# Isolate image name for center, left and right images
df_log.Center_Image = df_log.Center_Image.apply(lambda x: x.split('\\')[-1])
df_log.Left_Image = df_log.Left_Image.apply(lambda x: x.split('\\')[-1])
df_log.Right_Image = df_log.Right_Image.apply(lambda x: x.split('\\')[-1])

# Drop unnecessary columns
df_log.drop(columns=['Throttle', 'Brake', 'Speed'], inplace=True)

# Create a new dataframe for cleanup
df_new = pd.DataFrame()

# Set steering value for left and right images with correction
correction = 0.2
leftSteering = df_log.Steering + correction
rightSteering = df_log.Steering - correction

# Create a new image column with all center, left, right images
df_new['img_path'] = pd.concat([df_log.Center_Image, df_log.Left_Image, df_log.Right_Image], ignore_index=True)
# Create a steering column with corresponding measurement values
df_new['steering'] = pd.concat([df_log.Steering, leftSteering, rightSteering], ignore_index=True)

# Import Keras libraries
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# set batch size
batch_size = 32

# Split the dataset to training and validation samples
train_samples, validation_samples = train_test_split(df_new, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Create Model and add layers to it
# The model architecture used here is based on NVIDIA article:
# End-to-end Deep Learning for Self-Driving Cars
# https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
model = Sequential()
model.add(Lambda(lambda x: ((x/255.0) - 0.5), input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Convolution2D(36, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Convolution2D(48, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Convolution2D(64, kernel_size=(3,3), activation='relu'))
model.add(Convolution2D(64, kernel_size=(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Compile the model
model.compile(loss='mse', optimizer='adam')

checkpointpath = 'tmp'
# Set the callbacks
my_callbacks=  [
    EarlyStopping(monitor = 'val_loss', patience=3),
    ModelCheckpoint(filepath=checkpointpath, monitor='val_loss', save_best_only=True)
]
                           
# Train model with generator
history_object = model.fit(x=train_generator, \
                           validation_data=validation_generator, \
                           epochs=10, \
                           verbose=1, \
                           validation_steps=ceil(len(validation_samples) * 2/batch_size), \
                           steps_per_epoch=ceil(len(train_samples) * 2/batch_size), \
                           callbacks=my_callbacks)


plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.png', bbox_inches='tight')

# Save the model
model.save('model.h5')
