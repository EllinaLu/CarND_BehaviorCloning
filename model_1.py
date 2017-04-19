import csv
import cv2
import numpy as np
import h5py
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D as Conv2D

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_integer('epoch', 5, 'Train epoch num')
flags.DEFINE_string('out', 'model.h5', 'Out model file name')

drivingDataList = [
'../../Proj3_Reference/proj3_sample_train_data/data/driving_log.csv',
'../../Proj3_Reference/recorded_data/1/driving_log.csv',
'../../Proj3_Reference/recorded_data/2_bridge/driving_log.csv']

lines = []
#sample training data given by instructuor
for i in range(len(drivingDataList)):
    with open(drivingDataList[i]) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

print(len(lines), 'of samples')
images = []
measurements = []
for line in lines:
    path = line[0]
    image = cv2.imread(path)
    images.append(image)    
    measurement = float(line[3])
    measurements.append(measurement)
    
    #flip image & measurement to augment data
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurements.append(-measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

print('image shape= ', X_train[0].shape)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50, 10), (0, 0))))
model.add(Conv2D(16, 5, 5, activation='relu'))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, nb_epoch=FLAGS.epoch)

#model.save('model.h5')
model.save(FLAGS.out)

import gc
gc.collect()