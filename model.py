import csv
import cv2
import numpy as np
import h5py
import sklearn
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D as Conv2D

batch_size = 128
#define input parameters to the script: epoch, out file name
flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_integer('epoch', 5, 'Train epoch num')
flags.DEFINE_string('out', 'model.h5', 'Out model file name')

#list of data to use for training and validation
drivingDataList = [
'../../Proj3_Reference/proj3_sample_train_data/data/driving_log.csv',
'../../Proj3_Reference/recorded_data/1/driving_log.csv',
'../../Proj3_Reference/recorded_data/2_bridge/driving_log.csv',
#for track3
'../../Proj3_Reference/recorded_data/3_track2/driving_log.csv',
'../../Proj3_Reference/recorded_data/4_recovery/driving_log.csv',
'../../Proj3_Reference/recorded_data/5_tk1_backwards/driving_log.csv'] 

lines = []
#read in the location of the input image data and steer data from above array
for i in range(len(drivingDataList)):
    with open(drivingDataList[i]) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

#randomly split data into training portion (80%) and validation portion (20%)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print(len(lines), 'of samples')

#generator function to return a subset of the input data for each loop (yield)
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                #flip image & measurement to augment data
                image_flipped = np.fliplr(center_image)
                images.append(image_flipped)
                angles.append(-center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield   sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

print('train samples=', len(train_samples))

model = Sequential()
#normalize input image
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# crop image to only see section with road
model.add(Cropping2D(cropping=((60, 20), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Flatten())
#add dropout to avoid overfitting
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

#compile with mse error function and adam optimizer
model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, nb_epoch=FLAGS.epoch)
model.fit_generator(train_generator, steps_per_epoch = len(train_samples)/batch_size, validation_data=validation_generator,\
 validation_steps=len(validation_samples)/batch_size, nb_epoch=FLAGS.epoch)

#save trained model in the specified output file name
model.save(FLAGS.out)

import gc
gc.collect()