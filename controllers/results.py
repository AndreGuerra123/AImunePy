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
MODELS = {
    'host': 'localhost',
    'port': 27017,
    'database': 'authentication',
    'collection': 'models'
}

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

class Resulter:
    def __init__(self,params):
       self.model_id = toObjectId(params, 'source')
       models = connect(MODELS)
       model = models.find_one({'_id':self.model_id},{'results':1})
       self.result_id = toObjectId(model,'results')
       disconnect(models)
       self.history = LoadHistory(self.result_id,DATABASE)

    def getHtml(self):
        return PlotHistory(self.history,width=800,height=600)