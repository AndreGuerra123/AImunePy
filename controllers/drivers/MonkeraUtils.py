from keras.layers import Dense, Input
from keras.models import clone_model, Model, Sequential, model_from_json
from keras.callbacks import Callback
from pymongo import MongoClient
from bson.objectid import ObjectId
import pydash as p_
import base64
import time
import tempfile
import gridfs
import io
import os
import pickle

def _get(obj,loc):
    return p_.get(obj,loc)

def _getSafe(obj,loc,typ,msg):
    got = _get(obj,loc)
    assert isinstance(got,typ), msg
    return got

def connect(obj):
    return MongoClient(
        _getSafe(obj, 'host', str, 'Hostname is not a valid string.'),
        _getSafe(obj, 'port', int, 'Port is not a valid integer.'))[_getSafe(obj, 'database', str, 'Database name is not a valid string.')][_getSafe(obj, 'collection', str, 'Collection name is not a valid string.')]

def disconnect(collection):
    del collection
    collection = None

def ValidateModelArchitecture(model,inp,out):

    def validation(model,inp,out):
        assert isinstance(model,int) or isinstance(model,Model)
        n_in = len(model.inputs)
        n_out =len(model.outputs) 
        assert (n_in) > 0, 'Model has not detectable inputs.'
        assert (n_out) > 0, 'Model has not detectable outputs.'
        assert (n_in) <= 1, 'Model has multiple %d inputs tensors. Cannot apply input transformation.' % (n_in)
        assert (n_out) <= 1, 'Model has multiple %d output tensors Cannot apply output transformation.' % (n_out)
    
        inp_old = model.input_shape
    
        assert len(inp_old) == 4, 'Model input tensor shape != 4: Not a valid image classification model (B x X x X x X).'
    
        assert isinstance(inp,tuple), 'Input parameter is not a valid tuple.'
        assert len(inp) == 4, 'Input parameter is not a valid 4-rank tensor shape.'

        out_old = model.output_shape
        assert len(out_old) == 2, 'Model output tensor shape !=2: Not a valid image classification model (B x C).'
        assert isinstance(out,tuple), 'Output parameter is not a valid tuple.'
        assert len(out) == 2, 'Output parameter is not a valid 2-rank tensor shape.'

        ci = any([inp[i] != inp_old[i] for i in range(0,len(inp))])
        co = any([out[i] != out_old[i] for i in range(0,len(out))])
      
        return ci,co

    def findPreTop(model):
        i = len(model.layers)-1
        cos = model.output_shape
        while(model.layers[i].output_shape == cos):
            i -= 1
        return i
 
    def reshapeOutput(model,i,out): # Reshapes model accordingly to https://keras.io/applications/#usage-examples-for-image-classification-models
        return Dense(out[1], activation='softmax',name = 'output_aimune_'+str(int(round(time.time() * 1000))))(model.layers[i].output)
     
    def changeInp(model,inp):
        return clone_model(model,Input(batch_shape=inp, name = 'input_aimune_'+str(int(round(time.time() * 1000)))))
        
    def changeOut(model,out):
        idx = findPreTop(model) # Finds the pre-topping layer (must be tested more extensively)
        preds = reshapeOutput(model,idx,out)
        return Model(inputs=model.input, outputs=preds)

    ci,co = validation(model,inp,out)
        
    if(ci): #change input
        model = changeInp(model,inp)

    if(co): #change ouput
        model = changeOut(model,out)


    return model, any([ci,co])

def LoadModelArchitecture(query,location,connection={'host':'localhost','port':27017,'database':'database','collection':'collection'}):
    assert (type(query)==dict and bool(query)), 'Please provide a valid dictionary identifying which model architecture to retrieve. Ex. "{"_id":xxx}"'
    assert type(location) == str, 'Please provide a valid location of the model architecture.'
    assert type(connection) == dict, 'Please provide a valid connection dictionary.' 
    
    col = connect(connection)
    doc = col.find_one(query)
    disconnect(col)
    arch = _get(doc,location)

    if(isinstance(arch,str)):
        return _to_model(arch)
    elif(isinstance(arch,bytes)):
        arch2 = arch.decode()
        return _to_model(arch2)
    elif(arch is None):
        raise ValueError('Could not find an architecture object in the database.')
    else:
        raise TypeError('Detected architecture object is neither a string or binary json object.')

def _to_model(archi):
    assert type(archi)==str, 'Retrieved field (decoded) is not a valid Keras model json string.' 
    try:
            return model_from_json(archi) #simply encoded as binary or loaded as pure string
    except:           
            return model_from_json(base64.b64decode(archi).decode()) # the string was encoded in base64

class MonkeraCallback(Callback):
    def __init__(self,query,config={'ini':0,'end':1,'epochs':1,'value':'value','description':'description'},connection={'host':'localhost','port':27017,'database':'database','collection':'collection'}):
         super(MonkeraCallback, self).__init__()
         assert type(query) == dict and bool(query), 'Please insert a valid dictionary for the query parameter. Ex:"{"_id":xxx}"'
         self.query = query
         self.ini = _getSafe(config,'ini',(int,float),'Initialization parameter must be an integer or a float.')
         self.end = _getSafe(config,'end',(int,float),'Ending parameter must be an integer or a float.')
         assert self.end > self.ini ,'Ending parameter must be bigger than initialization parameter.'
         assert self.ini>=0 and self.ini<1, 'Initialization parameter must be between 0 inclusive and 1 exclusive.'
         assert self.end>0 and self.ini<=1, 'Ending parameter must be between 0 exclusive and 1 inclusive.'

         self.valuepath = _getSafe(config,'value',str, 'Value parameter must be a valid string referencing the path of the value to write in the document.')
         self.descriptionpath = _getSafe(config,'description',str, 'Description parameter must be a valid string referencing the path of the description to write in the document.')

         self.epochs = _getSafe(config,'epochs',int,'Epochs parameter must be a valid integer value.')
         assert self.epochs > 0, 'Epochs parameter must be positive.'

         _getSafe(connection,'host',str,'Host must be a valid string referencing to mongodb host.')
         _getSafe(connection,'port',int,'Port must be a valid integer referencing to mongodb port.')
         _getSafe(connection,'database',str,'Database must be a valid string referencing to mongodb database.')
         _getSafe(connection,'collection',str,'Collection must be a valid string referencing to mongodb collection.')
         self.connection = connection

    def on_epoch_end(self, epoch, logs = None):

            value = self.ini + (epoch/self.epochs)*(self.end-self.ini)
            description = "Fitting model: Epoch: "+str(epoch)+ " Loss: " + str(logs.get('loss'))
            col = connect(self.connection)
            col.update_one(self.query,{"$set":{self.valuepath:value,
                self.descriptionpath:description}})
            disconnect(col)

def SaveModelWeights(model,connection={'host':'localhost','port':27017,'database':'database'}):
    db = MongoClient(_getSafe(connection,'host',str,'Host must be a valid string referencing to mongodb host.'),
         _getSafe(connection,'port',int,'Port must be a valid integer referencing to mongodb port.'))[_getSafe(connection,'database',str,'Database must be a valid string referencing to mongodb database.')]
    try:
        fs = gridfs.GridFS(db)
        fd, name = tempfile.mkstemp()
        model.save_weights(name)
        return fs.put(open(fd, 'rb'))
    finally:
        os.remove(name)
        disconnect(db)

def LoadModelWeights(model,fileID,connection={'host':'localhost','port':27017,'database':'database'}):
    db = MongoClient(_getSafe(connection,'host',str,'Host must be a valid string referencing to mongodb host.'),
         _getSafe(connection,'port',int,'Port must be a valid integer referencing to mongodb port.'))[_getSafe(connection,'database',str,'Database must be a valid string referencing to mongodb database.')]
    assert isinstance(fileID,ObjectId) , 'Provided fileID parameter is not a valid ObjectID.'
    try:
        fs = gridfs.GridFS(db)
        out = fs.get(fileID)
        model.load_weights(out)
    finally:
        os.remove(out)
        disconnect(db)

def SaveHistory(history,connection={'host':'localhost','port':27017,'database':'database'}):
        db = MongoClient(_getSafe(connection,'host',str,'Host must be a valid string referencing to mongodb host.'),
         _getSafe(connection,'port',int,'Port must be a valid integer referencing to mongodb port.'))[_getSafe(connection,'database',str,'Database must be a valid string referencing to mongodb database.')]        
        try:
            fs = gridfs.GridFS(db)
            fd, name = tempfile.mkstemp()
            pickle.dump(history, fd)
            return fs.put(open(fd, 'rb'))
        finally:
            os.remove(name)
            disconnect(db)

def LoadHistory(fileID,connection={'host':'localhost','port':27017,'database':'database'}):
    db = MongoClient(_getSafe(connection,'host',str,'Host must be a valid string referencing to mongodb host.'),
         _getSafe(connection,'port',int,'Port must be a valid integer referencing to mongodb port.'))[_getSafe(connection,'database',str,'Database must be a valid string referencing to mongodb database.')]
    assert isinstance(fileID,ObjectId) , 'Provided fileID parameter is not a valid ObjectID.'
    try:
        fs = gridfs.GridFS(db)
        out = fs.get(fileID)
        return pickle.load(out)
    finally:
        os.remove(out)
        disconnect(db)
        



            

    