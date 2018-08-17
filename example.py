from controllers.drivers.Monkera import MongoImageDataGenerator

import keras
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras import layers
from keras.applications import InceptionV3
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

import json
import pydash as p_

""" mongogen = MongoImageDataGenerator(
                          connection={'host': "localhost", 'port': 27017,'database': "authentication", 'collection': "loads"},
                          query={},
                          location={'image': "image.data", 'label': "classi"},
                          config={'batch_size': 2, 'shuffle': True, 'seed': 123, 'width': 50, 'height': 50, 'data_format': 'channels_last',
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
                          
traingen, valgen = mongogen.flows_from_mongo() """

model = InceptionV3()
print(model.inputs)
print(model.outputs)

shape=(32,32,3)
stringmodel = model.to_json()


json_model=json.loads(stringmodel)

json_model['config'][0]['config']['input_shape'] = shape
json_model['config'][0]['config']['batch_input_shape'] = (None,)+shape

json_model['config'][-1]['config']['input_shape'] = (6)

model = model_from_json(json.dumps(json_model))



#model.layers[0].input.set_shape((None,)+mongogen.getShape())
#model.layers[len(model.layers)-1].output.set_shape((None,mongogen.getClassNumber()))
# initiate RMSprop optimizer"""
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6) 

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


trainflow,testflow = mongogen.flows_from_mongo()

model.fit_generator(traingen, epochs=10,validation_data=valgen, workers=4, use_multiprocessing=True)