from pymongo import MongoClient
from bson.objectid import ObjectId
import pydash as p_
from controllers.drivers.MonkeraUtils import LoadHistory, PlotHistory
from pymongo import MongoClient

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
        self.modelid = toObjectId(params,'model_id')
        self.tempid = toObjectId(params,'temp_id')
    def getPredictions(self):
        return False