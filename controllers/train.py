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

IMAGES = {
    'host': 'localhost',
    'port': 12721,
    'database':'authentication',
    'collection':'loads'
}

LOCATION = {
    'image': 'image.data',
    'label': 'classi'
}

MODELS = {
    'host': 'localhost',
    'port': 12721,
    'database':'authentication',
    'collection':'models'
}

JOBS = {
    'host': 'localhost',
    'port': 12721,
    'database':'authentication',
    'collection':'jobs'
}

def getSafe(obj,loc,typ,msg):
    tr = p_.get(obj,loc)
    assert tr != None, msg
    assert type(tr)==typ, msg
    return tr

def validateID(params, loc, msg):
    mongoID = getSafe(params,loc,str,msg)
    return ObjectID(mongoID)   
    
def connect(obj):
    return pymongo.MongoClient(obj['host'],obj['port'])[obj['database']]['collection']
    
def disconnect(collection):
    del collection
    collection = None

def includeAll(obj,location):
    return "-1" in p_.get(obj,location) 

class Trainer:
    def __init__(self,params):
        print(params)
        self.model_id = validateID(params,'model_id', "Model_ID is not a valid MongoDB ID string.")
        self.job_id = validateID(params,'job_id', "Job_ID is not a valid MongoDB ID string.")

        #Retrieving modelling parameters
        self.updateProgress(0,"Retrieving model parameters...")
        self.model_doc = self.getModelParameters()

        #Validating modelling parameters
        self.updateProgress(0.05,"Validating model parameters...")
        self.model_postdoc = parameterValidation(self.model_doc)
        
        #Creating Query
        self.updateProgress(0.1,"Setting image query parameter...")
        self.query = self.getQuery(self.model_postdoc)

        #Creating Image Data Flow Generators
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

        #Loading Architecture
        self.updateProgress(0.2,"Loading model architecture...")
        self.model = loadArchitecture()

        #Compiling Architecture
        self.updateProgress(0.25,"Compiling model loss, optimiser and metrics...") 
        self.model.compile(loss=get(self.model_postdoc,'batch_size'),
              optimizer=get(self.model_postdoc,'optimizer'),
              metrics=get(self.model_postdoc,'batch_size'))


        #Train Model
        self.updateProgress(0.3,"Retrieving model parameters and architecture...")
        self.model = self.model.fit_generator(self.traingen, epochs=get(self.model_postdoc,'epochs'), validation_data=self.valgen, workers=4, use_multiprocessing=True)

        #Save weigths
        self.updateProgress(0.9,"Saving model trained weights...")


        #Save results
        self.updateProgress(0.95,"Saving model results...")

    def updateProgress(self,value,strmsg):
        progress={'value':value,
                  'description':strmsg}
        col = connect(jobs)
        col.update_one({'_id':self.job_id},{'progress':progress})
        disconnect(col)

    def getModelParameters(self):
        col = connect(MODELS)
        toreturn = col.find({'_id':self.model_id,},{'dataset':1,'config':1,'architecture':1,'queue':1})
        disconnect(col)
        return toreturn

    def getQuery(self):
        query = {}
        if (not includeAll(self.model_doc,'dataset.patients')):
            query["patients"] = {'$in': params.dataset.patients}
        if (not includeAll(self.model_doc,'dataset.conditions')):
            query["conditions"] = {'$in': params.dataset.conditions}
        if (not includeAll(self.model_doc,'dataset.compounds')):
            query["compounds"] = {'$in': params.dataset.compounds}
        if (not includeAll(self.model_doc,'dataset.classes')):
            query["classes"] = {'$in': params.dataset.classes}
        return query
    
    def loadArchitecture(self):
        model = Models.model_from_json(json(get(self.model_doc,'architecture')))
        model.layers[0].input.set_shape(self.model_doc.getShape())
        model.layers[len(model.layers)].output.set_shape(self.mifg.getClassNumber)
        return model