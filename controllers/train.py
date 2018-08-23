from pymongo import MongoClient
from bson.objectid import ObjectId
import keras
import random
import json
from controllers.drivers.Monkera import MongoImageDataGenerator
from keras.models import Model, model_from_json
from keras.layers import deserialize, deserialize_keras_object
from PIL import Image
import numpy as np
import pydash as p_
from datetime import datetime

LOCATION = {
    'image': 'image.data',
    'label': 'classi'
}

MODELS = {
    'host': 'localhost',
    'port': 27017,
    'database': 'authentication',
    'collection': 'models'
}

def get(obj, loc):
    return p_.get(obj, loc)


def getSafe(obj, loc, typ, msg):
    tr = p_.get(obj, loc)
    assert tr != None, msg
    assert type(tr) == typ, msg
    return tr


def toObjectId(params, loc):
    obj = get(params, loc)
    if (isinstance(obj, ObjectId)):
        return obj
    elif (isinstance(obj, str)):
        return ObjectId(str(obj))
    else:
        raise ValueError(
            'Supplied object is not a valid ObjectId object or string')

def connect(obj):

    return MongoClient(
        getSafe(obj, 'host', str, 'Hostname is not a valid string.'),
        getSafe(obj, 'port', int, 'Port is not a valid integer.'))[getSafe(obj, 'database', str, 'Database name is not a valid string.')][getSafe(obj, 'collection', str, 'Collection name is not a valid string.')]

def disconnect(collection):
    del collection
    collection = None

def includeAll(obj, location):
    return "-1" in p_.get(obj, location)


class Trainer:

    def startJob(self):
        self.job_id = ObjectId()
        col = connect(MODELS)
        modelinit = col.find_one({"_id": self.model_id})
        dataset_date = getSafe(modelinit, 'dataset.date', datetime,
                               "Failed to retrieve the dataset configuration synchronisation date.")
        config_date = getSafe(modelinit, 'config.date', datetime,
                              "Failed to retrieve the model configuration synchronisation date.")
        self.file = {
            'job': {
                '_id': self.job_id,
                'started': datetime.utcnow(),
                'finished': None,
                'error': None,
                'value': 0,
                'description': "Started new Job..."
            },
            'sync': {
                'dataset_date': dataset_date,
                'config_date': config_date
            }}
        count = col.update_one({'_id': self.model_id}, {'$set': {
            'file': self.file
        }}).modified_count
        disconnect(col)
        if(count != 1):
            raise ValueError('Failed to create new job.')

    def finishJob(self):
        if(self.proceed()):
            self.file['job']['value'] = 1
            self.file['job']['description'] = "Job completed successfully."
            self.file['job']['finished'] = datetime.utcnow()

            col = connect(MODELS)
            col.update_one({'_id': self.model_id}, {'$set':
                                                    {'file': self.file}})
            disconnect(col)
        else:
            raise ValueError('Model training was canceled/reset.')

    # Returns False if cannot update meaning that it was canceled
    def updateProgress(self, value, strmsg):
        if(self.proceed()):
            self.file['job']['value'] = value
            self.file['job']['description'] = strmsg
            col = connect(MODELS)
            col.update_one({'_id': self.model_id}, {'$set': {
                'file': self.file}})
            disconnect(col)
        else:
            raise ValueError('Modelling job was canceled/reset.')

    def proceed(self):
        col = connect(MODELS)
        model = col.find_one({'_id': self.model_id}, {'file.job': 1})
        disconnect(col)
        if(model is None):
            return False
        elif(get(model, 'file.job._id') != self.job_id):
            return False
        elif(get(model, 'file.job.finished') is not None):
            return False
        else:
            return True

    def processError(self, errormsg):
        if(self.proceed()):
            self.file['job']['error'] = errormsg
            self.file['job']['finished'] = datetime.utcnow()
            col = connect(MODELS)
            col.update_one({'_id': self.model_id}, {'$set': {
                'file': self.file}})
            disconnect(col)
        else:
            pass

    def getModelParameters(self):
        col = connect(MODELS)
        toreturn = col.find_one({'_id': self.model_id, }, {
                            'dataset': 1, 'config': 1, 'architecture': 1})
        disconnect(col)
        return toreturn

    def getQuery(self):
        query = {}
        if (not includeAll(self.model_doc, 'dataset.patients')):
            query["patients"] = {'$in': params.dataset.patients}
        if (not includeAll(self.model_doc, 'dataset.conditions')):
            query["conditions"] = {'$in': params.dataset.conditions}
        if (not includeAll(self.model_doc, 'dataset.compounds')):
            query["compounds"] = {'$in': params.dataset.compounds}
        if (not includeAll(self.model_doc, 'dataset.classes')):
            query["classes"] = {'$in': params.dataset.classes}
        return query

    def loadModelArchitecture(self, model_doc):
        arch = getSafe(model_doc,'architecture.file',dict,'Failed to load the model architecture object.')
        return deserialize(arch)
        

    def parameterValidation(model):
        toreturn = {}
        getSafe()


    def __init__(self, params):

        self.model_id = toObjectId(params, 'source')
        self.startJob()
        try:
            # Retrieving modelling parameters
            self.updateProgress(0,"Retrieving model parameters...")
            self.model_doc = self.getModelParameters()

            # Validating modelling parameters
            self.updateProgress(0.05,"Loading model architecture...")
            self.model = self.loadModelArchitecture(self.model_doc)
        
            # Creating Query
            self.updateProgress(0.1,"Building image database query...")
            self.query = self.buildingQuery(self.model_doc)

            # Creating Image Data Flow Generators
            self.updateProgress(0.15,"Setting MongoDB image data generators...")
            self.mifg = MongoImageFlowGenerator(connection=IMAGES,
                 query=self.query,
                 location=LOCATION,
                 config={
                     'batch_size': get(self.model_postdoc,'batch_size'),
                     'shuffle': get(self.model_postdoc,'shuffle'),
                     'seed': get(self.model_postdoc,'seed'),
                     'target_size': get(self.model_postdoc,'target_size'),
                     'data_format': get(self.model_postdoc,'data_format'),
                     'color_format': get(self.model_postdoc,'color_format'),
                     'validation_split': get(self.model_postdoc,'validation_split')},
                 stand={
                     'center': get(self.model_postdoc,'shuffle'),
                     'normalize': get(self.model_postdoc,'normalize'),
                     'rescale': get(self.model_postdoc,'rescale'),
                     'preprocessing_function': None,
                 },
                 affine={
                     'rounds': get(self.model_postdoc,'rounds'),
                     'transform': get(self.model_postdoc,'transform'),
                     'random': get(self.model_postdoc,'random'),
                     'keep_original': get(self.model_postdoc,'keep_original'),
                     'rotation': get(self.model_postdoc,'rotation'),
                     'width_shift': get(self.model_postdoc,'width_shift'),
                     'height_shift': get(self.model_postdoc,'height_shift'),
                     'shear': get(self.model_postdoc,'shear'),
                     'channel_shift': get(self.model_postdoc,'channel_shift'),
                     'brightness': get(self.model_postdoc,'brightness'),
                     'zoom': get(self.model_postdoc,'zoom'),
                     'horizontal_flip': get(self.model_postdoc,'horizontal_flip'),
                     'vertical_flip': get(self.model_postdoc,'vertical_flip'),
                     'fill_mode': get(self.model_postdoc,'fill_mode'),
                     'cval': get(self.model_postdoc,'cval')
                 })
            self.traingen , self.valgen = self.mifg.flows_from_mongo()

            # Compiling Architecture
            self.updateProgress(0.25,"Compiling model loss, optimiser and metrics...") 
            self.model.compile(loss=get(self.model_postdoc,'batch_size'),optimizer=get(self.model_postdoc,'optimizer'))

            # Train Model
            self.updateProgress(0.3,"Retrieving model parameters and architecture...") 
            self.model = self.model.fit_generator(self.traingen, epochs=get(self.model_postdoc,'epochs'), validation_data=self.valgen, workers=4, use_multiprocessing=True)

            # Save weigths
            self.updateProgress(0.9,"Saving model trained weights...") 

            # Save results
            self.updateProgress(0.95,"Saving model results...") 
    
            self.finishJob() 

        except Exception as e:
            self.processError(str(e))
