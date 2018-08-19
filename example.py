from controllers.drivers.Monkera import MongoImageDataGenerator
from controllers.drivers.MonkeraUtils import Modify

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.applications import InceptionV3
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

import json
import pydash as p_

mongogen = MongoImageDataGenerator(
                          connection={'host': "localhost", 'port': 27017,'database': "authentication", 'collection': "loads"},
                          query={},
                          location={'image': "image.data", 'label': "classi"},
                          config={'batch_size': 2, 'shuffle': True, 'seed': 123, 'target_size': (50,50), 'data_format': 'channels_last',
                            'color_format': 'RGB',
                            'validation_split': 0.5},
                          stand={
                            'preprocessing_function': None,
                            'rescale': 1/255,
                            'center': True,
                            'normalize': True
                            },
                          affine={
                            'rounds': 1,
                            'transform': True,
                            'random': False,
                            'keep_original': False,
                            'rotation': 0.1,
                            'width_shift': 0.2,
                            'height_shift': 0.3,
                            'shear': 0.1,
                            'channel_shift': 0.1,
                            'brightness': 0.4,
                            'zoom': 0.1,
                            'horizontal_flip': False,
                            'vertical_flip': False,
                            'fill_mode': 'nearest',
                            'cval': 0.
                            }
                          )
model = InceptionV3()                       
""" model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape = (100,100,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5,name="damn"))
model.add(Dense(2))
model.add(Activation('softmax')) """

traingen, valgen = mongogen.flows_from_mongo()
model, modified = Modify(model,mongogen.getInputShape(),mongogen.getOutputShape())

# initiate RMSprop optimizer"""
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6) 

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit_generator(traingen, epochs=10,validation_data=valgen, workers=4, use_multiprocessing=True)