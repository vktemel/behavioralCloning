# Load GPU driver

#from tensorflow.python.client import device_lib 
#print(device_lib.list_local_devices())


import pandas as pd
import numpy as np
import cv2

drive_log = 'data/driving_log.csv'
df_log = pd.read_csv(drive_log, names=['Center_Image', 'Left_Image', 'Right_Image', 'Steering', 'Throttle', 'Brake', 'Speed'])

print(df_log.shape[0])
print(df_log.Center_Image[0].split('\\')[-1])

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

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape = (160,320,3)))
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
