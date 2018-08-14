from controllers.drivers.Monkera2 import MongoImageDataGenerator

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D



datagen = MongoImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=True,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=0,
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=1/255,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format="channels_last",
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.1)

mongogen = MongoGenerator(connection={'host': "localhost", 'port': 27017,'database': "authentication", 'collection': "loads"},
                          query={},
                          location={'image': "image.data", 'label': "classi"},
                          config={'batchsize': 2, 'shuffle': True, 'seed': 123, 'width': 50, 'height': 50},
                          )

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

model.fit_generator(trainflow, epochs=3,validation_data=testflow, workers=4)


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
