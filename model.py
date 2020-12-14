import pandas as pd
import numpy as np
import cv2

drive_log = 'data/driving_log.csv'
df_log = pd.read_csv(drive_log, names=['Center_Image', 'Left_Image', 'Right_Image', 'Steering', 'Throttle', 'Brake', 'Speed'])

images = []
measurements = []
for datapoint in range(df_log.shape[0]):
    imgName = df_log.Center_Image[datapoint].split('\\')[-1]
    imgPath = 'data\\IMG\\' + imgName
    images.append(cv2.imread(imgPath))
    measurements.append(df_log.Steering[datapoint])

    correction = 0.2
    imgName = df_log.Left_Image[datapoint].split('\\')[-1]
    imgPath = 'data\\IMG\\' + imgName
    images.append(cv2.imread(imgPath))
    measurements.append(df_log.Steering[datapoint] + correction)

    imgName = df_log.Right_Image[datapoint].split('\\')[-1]
    imgPath = 'data\\IMG\\' + imgName
    images.append(cv2.imread(imgPath))
    measurements.append(df_log.Steering[datapoint]-correction)

augmented_images, augmented_measurements = [], []
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

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

model.compile(loss='mse', optimizer='adam')

history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 5)

model.save('model.h5')

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
