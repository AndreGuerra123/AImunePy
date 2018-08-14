from controllers.drivers.Monkera import MongoImageDataGenerator

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D



mongogen = MongoImageDataGenerator(
                          connection={'host': "localhost", 'port': 27017,'database': "authentication", 'collection': "loads"},
                          query={},
                          location={'image': "image.data", 'label': "classi"},
                          config={'batch_size': 2, 'shuffle': True, 'seed': 123, 'width': 50, 'height': 50, 'data_format': 'channels_last',
                            'color_format': 'RGB',
                            'validation_split': 0.},
                          stand={
                            'preprocessing_function': None,
                            'rescale': 1/255,
                            'center': False,
                            'normalize': True
                            },
                          affine={
                            'rounds': 0,
                            'transform': True,
                            'random': True,
                            'keep_original': False,
                            'rotation': 0.,
                            'width_shift': 0.,
                            'height_shift': 0.,
                            'shear': 0.,
                            'channel_shift': 0.,
                            'brightness': 0.,
                            'zoom': 0.,
                            'horizontal_flip': False,
                            'vertical_flip': False,
                            'fill_mode': 'nearest',
                            'cval': 0.
                 }
                          
                          )
traingen, valgen = mongogen.flows_from_mongo()



model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape = mongogen.getShape()))
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
model.add(Dropout(0.5))
model.add(Dense(mongogen.getClassNumber()))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

trainflow,testflow = mongogen.flows_from_mongo()

model.fit_generator(traingen, epochs=10,validation_data=valgen, workers=4, use_multiprocessing=True)


""" from flask import Flask, request, jsonify
from flask_cors import CORS
from controllers.train import trainer
from controllers.predict import predictor
from controllers.evaluate import evaluator

app = Flask(__name__)
CORS(app)

# this will route the request to the an API wich performs modelling in tensorflow


@app.route("/train", methods=['POST'])
def train():
    return jsonify(trainer(request.json))

# this will route the request to the an API wich performs metrics evaluation in tensorflow


@app.route("/evaluate", methods=['POST'])
def evaluate():
    return jsonify(evaluator(request.json))

# this will route the request to the an API wich performs image prediciton in tensorflow


@app.route("/predict", methods=['POST'])
def predict():
    return jsonify(predictor(request.json))

# HTTP Errors handlers


@app.errorhandler(404)
def url_error(e):
    return jsonify(e)


@app.errorhandler(500)
def server_error(e):
    return jsonify(e)


# This will run this microserver in localhost port 5000

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
 """
