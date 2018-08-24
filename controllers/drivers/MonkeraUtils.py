from keras.layers import Dense, Input
from keras.models import clone_model, Model, Sequential, model_from_json
from pymongo import MongoClient
from bson.objectid import ObjectId
import pydash as p_
import base64

def _get(obj,loc):
    return p_.get(obj,loc)

def _getSafe(obj,loc,typ,msg):
    got = _get(obj,loc)
    assert type(got)==typ, msg
    return got

def connect(obj):
    return MongoClient(
        _getSafe(obj, 'host', str, 'Hostname is not a valid string.'),
        _getSafe(obj, 'port', int, 'Port is not a valid integer.'))[_getSafe(obj, 'database', str, 'Database name is not a valid string.')][_getSafe(obj, 'collection', str, 'Collection name is not a valid string.')]

def disconnect(collection):
    del collection
    collection = None

def toObjectId(got):
    if (isinstance(got, ObjectId)):
        return got
    elif (isinstance(got, str)):
        return ObjectId(str(got))
    else:
        raise ValueError(
            'Supplied object is not a valid ObjectId object or string')

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
        layer=model.layers[i]
        x = layer.output
        x = Dense(out[1], activation='softmax')(x)
        return x  

    def changeInp(model,inp):
        return clone_model(model,Input(batch_shape=inp))
        
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

def LoadModelFromDatabase(obid,location,connection={'host':'localhost','port':27017,'database':'database','collection':'collection'}, decode = True):
    assert type(location)== str, 'Please provide a valid location of the model architecture.'
    obid = toObjectId(obid)
    col = connect(connection)
    doc = col.find_one({'_id':obid})
    disconnect(col)
    arch = _get(doc,location)
    if(isinstance(arch,str)):
        a = base64.b64decode(arch) if decode else arch
        print(a)
        return model_from_json(a)
    elif(arch is None):
        raise ValueError('Could not find an architecture object in the database.')
    else:
        raise TypeError('Detected architecture object is neither a string or binary json object.')

  
    