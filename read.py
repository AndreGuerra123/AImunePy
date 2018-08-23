from pymongo import MongoClient
import json
from keras.models import model_from_json
import io
import pydash as p_
  
col =  MongoClient('localhost',27017)['authentication']['architectures']
arch = col.find_one()
binary = p_.get(arch,'file')
print(binary)