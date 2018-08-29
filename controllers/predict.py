from pymongo import MongoClient
from bson.objectid import ObjectId
import pydash as p_
from controllers.drivers.MonkeraUtils import _to_model, LoadModelWeights, LoadModelArchitecture, ValidateImageModel
from pymongo import MongoClient
from keras import backend as K


DATABASE = {
    'host': 'localhost',
    'port': 27017,
    'database': 'authentication'
}

TEMPS = dict(DATABASE)
TEMPS['collection'] = 'temps'

MODELS = dict(DATABASE)
MODELS['collection'] = 'models'

def _get(obj, loc):
    return p_.get(obj, loc)

def _getSafe(obj, loc, typ, msg):
    tr = p_.get(obj, loc)
    assert isinstance(tr,typ), msg
    return tr

def connect(obj):
    return MongoClient(
        _getSafe(obj, 'host', str, 'Hostname is not a valid string.'),
        _getSafe(obj, 'port', int, 'Port is not a valid integer.'))[_getSafe(obj, 'database', str, 'Database name is not a valid string.')][_getSafe(obj, 'collection', str, 'Collection name is not a valid string.')]

def disconnect(collection):
    del collection
    collection = None

def toObjectId(params, loc):
    obj = _get(params, loc)
    if (isinstance(obj, ObjectId)):
        return obj
    elif (isinstance(obj, str)):
        return ObjectId(str(obj))
    else:
        raise ValueError(
            'Supplied object is not a valid ObjectId object or string')

class Predictor:
    def __init__(self, params):
        K.clear_session()
        self.modelid = toObjectId(params,'model_id')
        self.tempid = toObjectId(params,'temp_id')
        models = connect(MODELS)
        self.model_doc = models.find_one({'_id':self.modelid})
        disconnect(models)
        temps = connect(TEMPS)
        self.temp_doc = temps.find_one({'_id':self.tempid})
        disconnect(temps)

        self.model = LoadModelArchitecture({'_id':self.modelid},'architecture.file',MODELS)
        LoadModelWeights(self.model,_getSafe(self.model_doc,'weights',(str,ObjectId),'Could not find a valid weights model file.'),DATABASE)
        self.hotlabels = _getSafe(self.model_doc,'hotlabels',dict,'Could not find a valid hot labels dictionary.')

        self.imagetensor = ValidateImageModel({'_id':self.tempid},'image.data',TEMPS,config={
            'target_size':(_getSafe(self.model_doc,'dataset.height',int,'Failed to retrieve target height integer.'),_getSafe(self.model_doc,'dataset.width',int,'Failed to retrieve target width integer.')),
            'data_format':_getSafe(self.model_doc,'dataset.data_format',str,'Failed to retrieve target data_format string.'),
            'color_format':_getSafe(self.model_doc,'dataset.color_format',str,'Failed to retrieve target color_format string.')
        })
        self.sampletensor = np.array([imagetensor,])

    def getPredictions(self):
        print(self.model.predict(self.sampletensor)