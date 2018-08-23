from pymongo import MongoClient
import json
from keras.models import model_from_json
import pydash as p_
import bson
import io
  
col =  MongoClient('localhost',27017)['authentication']['architectures']
arch = col.find_one()
binary = p_.get(arch,'file')
jsonstr = io.BytesIO(binary)
model = model_from_json()
print(model)