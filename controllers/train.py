import pymongo;
db = pymongo.MongoClient()["authentication"];
images = db["loads"];
model = db["models"];

from keras.models import model_from_json


def trainer(params):

    model = model_from_json(params.architecture);

    model.compile(loss=params.config.loss,
              optimizer=params.config.optimiser,
              metrics=params.config.metrics)



    return 'damn';