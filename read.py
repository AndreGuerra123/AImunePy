from pymongo import MongoClient
import json
from keras.models import model_from_json
import io
import pydash as p_
from bson import json_util
  
col =  MongoClient('localhost',27017)['authentication']['architectures']
arch = col.find_one()
binary = p_.get(arch,'file')
jsonstring = json_util.dumps(binary)
model = model_from_json(jsonstring)
print(model)