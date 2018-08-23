from pymongo import MongoClient
import json
from keras.models import model_from_json
import pydash as p_
import bson
import json
import io
  
col =  MongoClient('localhost',27017)['authentication']['architectures']
arch = col.find_one()
binary = p_.get(arch,'file')
jsonstr = json.dumps(io.BytesIO(binary))
print(jsonstr)
model = model_from_json(jsonstr)
print(model)