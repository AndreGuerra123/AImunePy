from pymongo import MongoClient
from bson.objectid import ObjectId
import keras
import random
import json
from controllers.drivers.Monkera import MongoImageDataGenerator
from controllers.drivers.MonkeraUtils import ValidateModelArchitecture, LoadModelArchitectureBase64
from keras.models import Model, model_from_json
from keras.layers import deserialize, deserialize_keras_object
from PIL import Image
import numpy as np
import pydash as p_
from datetime import datetime
import traceback
import base64
from bson import Binary


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

IMAGES = {
    'host': 'localhost',
    'port': 27017,
    'database': 'authentication',
    'collection': 'loads'
}

def get(obj, loc):
    return p_.get(obj, loc)


def getSafe(obj, loc, typ, msg):
    tr = p_.get(obj, loc)
    assert isinstance(tr,typ), msg
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

def toInclude(lista):
    return ("-1" not in lista)

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

    def loadModelArchitecture(self, model_doc):
        return LoadModelArchitectureBase64(self.model_id,'architecture.file',MODELS)

    def getDatabaseQuery(self,model_doc):
        query = {}
        querypatients = getSafe(model_doc,'dataset.patients',list,'Failed to retrieve patients list.')
        queryconditions = getSafe(model_doc,'dataset.conditions',list,'Falied to retrieve conditions list.')
        querycompounds = getSafe(model_doc,'dataset.compounds',list,'Failed to retrieve compounds list.')
        queryclasses = getSafe(model_doc,'dataset.classes',list,'Failed to retrieve classes list.')
        if (toInclude(querypatients)):
            query["patients"] = {'$in': querypatients}
        if (toInclude(queryconditions)):
            query["conditions"] = {'$in': queryconditions}
        if (toInclude(querycompounds)):
            query["compounds"] = {'$in': querycompounds}
        if (toInclude(queryclasses)):
            query["classes"] = {'$in': queryclasses}
        return query

    def createGenerators(self):
        self.mifg = MongoImageDataGenerator(connection=IMAGES,
                 query=self.query,
                 location=LOCATION,
                 config={
                     'batch_size': getSafe(self.model_doc,'config.batchsize',int,'Could not retrieve a valid batchsize parameter.'),
                     'shuffle': getSafe(self.model_doc,'config.shuffle',bool,'Could not retrieve a valid suffle parameter.'),
                     'seed': getSafe(self.model_doc,'config.seed',(type(None),int),'Could not retrieve a valid seed parameter.'),
                     'target_size': (getSafe(self.model_doc,'dataset.height',int,'Could not retrieve a valid heigh parameter.'),getSafe(self.model_doc,'dataset.width',int,'Could not retrieve a valid width parameter.')),
                     'data_format': getSafe(self.model_doc,'dataset.data_format',str,'Could not retrieve a valid data_format parameter.'),
                     'color_format': getSafe(self.model_doc,'dataset.color_format',str,'Could not retrieve a valid color_format parameter.'),
                     'validation_split': getSafe(self.model_doc,'config.validation_split',float,'Could not retrieve a valid validation_split parameter.')},
                 stand={
                     'center': getSafe(self.model_doc,'dataset.center',bool,'Could not retrieve valid center parameter.'),
                     'normalize': getSafe(self.model_doc,'dataset.normalise',bool,'Could not retrieve valid normalise parameter.'),
                     'rescale': getSafe(self.model_doc,'dataset.rescale',float,'Could not retrieve a valid rescale parameter.'),
                     'preprocessing_function': None,
                 },
                 affine={
                     'rounds': getSafe(self.model_doc,'dataset.rounds',int,'Could not get a valid rounds parameter.'),
                     'transform': getSafe(self.model_doc,'dataset.transform',bool,'Could not get a valid transform parameter.'),
                     'random': getSafe(self.model_doc,'dataset.random',bool,'Could not get a valid random parameter.'),
                     'keep_original': getSafe(self.model_doc,'dataset.keep',bool,'Could not get a valid keep_original parameter.'),
                     'rotation': getSafe(self.model_doc,'dataset.rotation',(int,float),'Could not get a valid rotation parameter.'),
                     'width_shift': getSafe(self.model_doc,'dataset.width_shift',(int,float),'Could not get a valid width shift parameter.'),
                     'height_shift': getSafe(self.model_doc,'dataset.height_shift',(int,float),'Could not get a valid height shift parameter.'),
                     'shear': getSafe(self.model_doc,'dataset.shear',(int,float),'Could not get a valid shear parameter.'),
                     'channel_shift': getSafe(self.model_doc,'dataset.channel_shift',(int,float),'Could not get a valid channel shift parameter.'),
                     'brightness': getSafe(self.model_doc,'dataset.brightness',(int,float),'Could not get a valid brightness parameter.'),
                     'zoom': getSafe(self.model_doc,'dataset.zoom',(int,float),'Could not get a valid zoom parameter.'),
                     'horizontal_flip': getSafe(self.model_doc,'dataset.horizontal_flip',bool,'Could not get a valid horizontal flip parameter.'),
                     'vertical_flip': getSafe(self.model_doc,'dataset.vertical_flip',bool,'Could not get a valid vertical flip parameter.'),
                     'fill_mode': getSafe(self.model_doc,'dataset.fill_mode',str,'Could not get a valid fill mode parameter'),
                     'cval': getSafe(self.model_doc,'dataset.cval',(int,float),'Could not get a valid cval parameter.')
                 })
        self.traingen , self.valgen = self.mifg.flows_from_mongo()

    def validateModelArchitecture(self):
        self.model, self.modified = ValidateModelArchitecture(self.model,self.mifg.getInputShape(),self.mifg.getOutputShape())
        if(self.modified):
                self.saveArchitecture()

    def saveArchitecture(self):
        col = connect(MODELS)
        arch = base64.b64encode(bytes(self.model.to_json(),'utf8'))
        type(arch)
        col.update_one({'_id':self.model_id},{'$set':{'architecture.file': arch}})
        disconnect(col)
               
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
            self.query = self.getDatabaseQuery(self.model_doc)

            # Creating Image Data Flow Generators
            self.updateProgress(0.15,"Setting MongoDB image data generators...")
            self.createGenerators()
            
            # Validating model90
            self.updateProgress(0.25,"Validating model architecture and generators...") 
            self.validateModelArchitecture()

            # Compiling Architecture
            self.updateProgress(0.3,"Compiling model loss, optimiser and metrics...") 
            self.compiling()

            # Train Model
            self.updateProgress(0.3,"Retrieving model parameters and architecture...") 
            self.model = self.model.fit_generator(self.traingen, epochs=get(self.model_doc,'epochs'), validation_data=self.valgen, workers=4, use_multiprocessing=True)

            # Save weigths
            self.updateProgress(0.9,"Saving model trained weights...") 

            # Save results
            self.updateProgress(0.95,"Saving model results...") 
    
            self.finishJob() 

        except Exception as e:
            print(traceback.format_exc())
            self.processError(str(e))
