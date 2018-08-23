from pymongo import MongoClient
import json
from io import StringIO
from keras.models import model_from_json
import pydash as p_


  
col =  MongoClient('localhost',27017)['authentication']['architectures']
arch = col.find_one()
binary = p_.get(arch,'file').decode()
bina = json.load(StringIO(binary))
print(bina)