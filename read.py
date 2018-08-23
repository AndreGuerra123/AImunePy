from pymongo import MongoClient
import json
from keras.models import model_from_json
import io
  
col =  MongoClient('localhost',27017)['authentication']['architectures']
print(col.find_one())
json = json.loads(col.find_one().file)

print(json)