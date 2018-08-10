import io
import pymongo
from PIL import Image

import tensorflow
import theano
import numpy as np
import pydash as p_

import keras
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

def _get(obj,loc):
        return p_.get(obj,loc);

def _ag(obj,loc, msg):
    pro = _get(obj,loc)
    assert pro != None, msg
    return pro

def _agt(obj,loc,typ,msg):
        pro = _get(obj,loc)
        assert pro != None, msg
        assert type(pro) is typ, msg
        return pro

def _ad(object,msg):
        assert (type(object) is dict),msg

class MongoGenerator(Iterator):
    def __init__(self,
                 image_data_generator,
                 connection={'host':"localhost",'port':12721,'database': "database", 'collection': "collection"},
                 query={},
                 location={'image': "image", 'label': "label"},
                 config={'batchsize':5,'shuffle':True,'seed':123,'width':50,'height':50}):
       
        # Validate inputs
        assert isinstance(image_data_generator,ImageDataGenerator), "Please provide a valid instance of ImageDataGenerator for data augmentation."
        self.image_data_generator = image_data_generator

        #Check if inputs are json deserialised objects
        _ad(connection,"Please select a valid connection dictionary")
        _ad(query,"Please select a valid query dictionary")
        _ad(location,"Please select a valid location dictionary")
        _ad(config,"Please select a valid config dictionary")

        #Check for type an load
        self._host = _agt(connection,'host',str,"Please provide a valid string for mongodb hostname.")
        self._port = _agt(connection,'port',int,"Please provide a valid integer for mongodb port.")
        self._database = _agt(connection,'database',str,"Please provide a valid string for mongodb database.")
        self._collection = _agt(connection,'collection',str,"Please provide a valid string for mongodb collection.")

        self._img_location = _agt(location,'image',str,"Please provide a valid location for the image binary field in the selected mongodb collection.")
        self._lbl_location = _agt(location,'image',str,"Please provide a valid location for the label field in the selected mongodb collection.")

        self._batchsize = _agt(config,'batchsize',int,"Please select a valid integer value for the batchsize parameter.")
        self._shuffle = _agt(config,'shuffle',bool,"Please select a valid boolean value for the shuffle parameter.")
        self._seed = _agt(config,'seed',int,"Please select a valid integer value for the seed parameter.")
        self._heigth = _agt(config,'height',int,"Please select a valid integer value for the image height parameter.")
        self._width = _agt(config,'width',int,"Please select a valid integer value for the image width parameter.")

        self._dtype = K.floatx()
        self._size = (self._heigth,self._width)
        self._object_ids = self.__getOBIDS(query)

        self._samples = len(self._object_ids)
        assert (self._samples > 0),"The resulted query returned zero(0) samples."
        assert (self._samples > self._batchsize),"The resulted query returned less samples than the selected batchsize."

        self._classes = self.__getClassNumber()
        assert (self._classes > 1),"The resulted query return insufficient distinct classes."

        super(MongoGenerator, self).__init__(self._samples, self._batchsize, self._shuffle, self._seed)

    def __getOBIDS(self,query):
        collection = self.__connect()
        object_ids = list(collection.find(query, {'_id': True}))
        self.__disconnect(collection)
        return object_ids

    def __getClassNumber(self):
        collection = self.__connect()
        classes = collection.distinct(self._lbl_location,{'_id':{'$in':self._object_ids}}).length
        print(classes)
        return classes

    def _get_batches_of_transformed_samples(self, index_array):

        batch_x = np.zeros((len(index_array),self._height,self._width,3), dtype=self._dtype)
        
        batch_y = np.zeros((len(index_array),self._classes), dtype=self._dtype)

        for i, j in enumerate(index_array):

            #Get sample data
            (x,y) = self.__readMongoSample(self._object_ids[j])

            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            batch_y[i] = y

            return batch_x, batch_y

    def next(self):
        return self._get_batches_of_transformed_samples(next(self.index_generator))

    def __readMongoSample(self, oid):

        collection = self.__connect()
        sample = collection.find_one({'_id': oid}).next()
        assert sample != None, "Failed to retrieve the sample corresponding to the image ID: "+str(oid)
        return (self.__getImage(sample),self.__getLabel(sample))

    def __connect(self):
        return pymongo.MongoClient(self._host,self._port)[self._database][self._collection]

    def __disconnect(self, collection):
        del collection
        collection = None

    def __getImage(self,sample):
        strg = _ag(sample, self._img_location,"Failed to retrieve image binary (ID:"+_get(sample,'_id').__str__+") at "+self._img_location+".");
        img = img.open(io.BytesIO(strg)).resize(self._size)
        return np.asarray(img, dtype=self._dtype)

    def __geLabel(self,label):
        strg = _ag(sample, self._img_location,"Failed to retrieve image label (ID:"+_get(sample,'_id').__str__+") at "+self._lbl_location+".");
        return keras.utils.to_categorical(label, self._classes)

    def getShape(self):
        return (self._heigth,self._width,3)
    
    def getClassNumber(self):
        return self._classes
        
    



