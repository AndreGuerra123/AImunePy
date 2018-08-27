from pymongo import MongoClient
from bson.objectid import ObjectId
import pydash as p_
from controllers.drivers.MonkeraUtils import LoadHistory, PlotHistory

DATABASE = {
    'host': 'localhost',
    'port': 27017,
    'database': 'authentication'
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

class Resulter:
    def __init__(self,params):
       self.model_id = toObjectId(params, 'source')
       models = connect(MODELS)
       model = models.find_one({'_id':self.model_id},{'results':1,'_id':1})
       self.result_id = toObjectId(getSafe(model,'results',(str,ObjectId),'Could not find a valid results id refering the GridFS holding the file chuncks.'))
       disconnect(models)
       self.history = LoadHistory(self.result_id,DATABASE)

    def getHtml():
        return PlotHistory(self.history,width=800,height=600)