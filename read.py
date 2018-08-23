from pymongo import MongoClient
import json
from io import StringIO
from keras.models import model_from_json
import pydash as p_


  
col =  MongoClient('localhost',27017)['authentication']['architectures']
arch = col.find_one()
decoded = p_.get(arch,'file').decode()
model = model_from_json(decoded)