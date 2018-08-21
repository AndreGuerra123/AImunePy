import pymongo
from bson.objectid import ObjectId
import keras
import random
import json
from controllers.drivers.Monkera import MongoImageDataGenerator
import keras.models as Models
from PIL import Image
import numpy as np
import pydash as p_
from datetime import datetime

IMAGES = {
    'host': '127.0.0.1',
    'port': 27017,
    'database': 'authentication',
    'collection': 'loads'
}

LOCATION = {
    'image': 'image.data',
    'label': 'classi'
}

MODELS = {
    'host': '127.0.0.1',
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


def str2ObjectId(params, loc, msg):
    mongoID = getSafe(params, loc, str, msg)
    return ObjectId(mongoID)


def connect(obj):
    return pymongo.MongoClient(obj['host'], obj['port'])[obj['database']]['collection']


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
            self.file['job']['value'] = 1;
            self.file['job']['description'] = "Job completed successfully.";
            self.file['job']['finished'] = datetime.utcnow()

            col = connect(MODELS)
            col.update_one({'_id': self.model_id}, {'$set':
            {'file':self.file}})
            disconnect(col)
        else:
            raise ValueError('Model training was canceled/reset.')

    # Returns False if cannot update meaning that it was canceled
    def updateProgress(self, value, strmsg):
        if(self.proceed()):
            self.file['job']['value'] = value;
            self.file['job']['description'] = strmsg;
            col = connect(MODELS)
            col.update_one({'_id': self.model_id}, {'$set': {
            'file': self.file}})
            disconnect(col)
        else:
            raise ValueError('Modelling job was canceled/reset.')

    def proceed(self):
        col = connect(MODELS)
        model = col.find_one({'_id': self.model_id}, {'file.job':1})
        disconnect(col)
        if(model is None):
            return False
        elif(get(model, 'file.job._id') != self.job_id):
            return False
        elif(get(model,'file.job.finished') is not None):
            return False 
        else:
            return True

    def processError(self, errormsg):
        if(self.proceed()):
            self.file['job']['error'] = errormsg
            self.file['job']['finished'] = datetime.utcnow()
            col = connect(MODELS)
            col.update_one({'_id': self.model_id}, {'$set': {
            'file':self.file}})
            disconnect(col)
        else:
            pass

    def getModelParameters(self):
        col = connect(MODELS)
        toreturn = col.find({'_id': self.model_id, }, {
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

    def loadArchitecture(self):
        model = Models.model_from_json(
            json(get(self.model_doc, 'architecture')))
        model.layers[0].input.set_shape(self.model_doc.getShape())
        model.layers[len(model.layers)].output.set_shape(
            self.mifg.getClassNumber)
        return model

    def __init__(self,params):

        self.model_id = str2ObjectId(params,'source', "Model source is not a valid MongoDB ID string.")
        col = connect(MODELS)
        print(col.find_one({'_id':self.model_id}))
        disconnect(col)
        """ self.startJob()
        try:
            # Retrieving modelling parameters
            self.updateProgress(0,"Retrieving model parameters...")
            self.model_doc = self.getModelParameters()

            # Validating modelling parameters
            self.updateProgress(0.05,"Validating model parameters...")
            self.model_postdoc = parameterValidation(self.model_doc)
        
            # Creating Query
            self.updateProgress(0.1,"Setting image query parameter...")
            self.query = self.getQuery(self.model_postdoc)

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

            # Loading Architecture
            self.updateProgress(0.2,"Loading model architecture...") 
            self.model = loadArchitecture()

            # Compiling Architecture
            self.updateProgress(0.25,"Compiling model loss, optimiser and metrics...") 
            self.model.compile(loss=get(self.model_postdoc,'batch_size'),
              optimizer=get(self.model_postdoc,'optimizer'),
              metrics=get(self.model_postdoc,'batch_size'))

            # Train Model
            self.updateProgress(0.3,"Retrieving model parameters and architecture...") 
            self.model = self.model.fit_generator(self.traingen, epochs=get(self.model_postdoc,'epochs'), validation_data=self.valgen, workers=4, use_multiprocessing=True)

            # Save weigths
            self.updateProgress(0.9,"Saving model trained weights...") 

            # Save results
            self.updateProgress(0.95,"Saving model results...") 
    
            self.finishJob() 

        except Exception as e:
            self.processError(str(e)) """

   
