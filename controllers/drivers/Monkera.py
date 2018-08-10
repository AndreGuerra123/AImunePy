import io
import pymongo
from PIL import Image

import tensorflow
import theano
import numpy as np
import pydash as p_
import collections
import scipy

import keras
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


def _get(obj, loc):
    return p_.get(obj, loc)


def _ag(obj, loc, msg):
    pro = _get(obj, loc)
    assert pro != None, msg
    return pro


def _agt(obj, loc, typ, msg):
    pro = _get(obj, loc)
    assert pro != None, msg
    assert type(pro) is typ, msg
    return pro


def _ad(object, msg):
    assert (type(object) is dict), msg


class MongoGenerator(Iterator):
    def __init__(self,
                 image_data_generator,
                 connection={'host': "localhost", 'port': 12721,
                             'database': "database", 'collection': "collection"},
                 query={},
                 location={'image': "image", 'label': "label"},
                 config={'batchsize': 5, 'shuffle': True, 'seed': 123, 'width': 50, 'height': 50}):

        # Validate inputs
        assert isinstance(
            image_data_generator, ImageDataGenerator), "Please provide a valid instance of ImageDataGenerator for data augmentation."
        self.image_data_generator = image_data_generator

        # Check if inputs are json deserialised objects
        _ad(connection, "Please select a valid connection dictionary")
        _ad(query, "Please select a valid query dictionary")
        _ad(location, "Please select a valid location dictionary")
        _ad(config, "Please select a valid config dictionary")

        # Check for type an load
        self._host = _agt(connection, 'host', str,
                          "Please provide a valid string for mongodb hostname.")
        self._port = _agt(connection, 'port', int,
                          "Please provide a valid integer for mongodb port.")
        self._database = _agt(connection, 'database', str,
                              "Please provide a valid string for mongodb database.")
        self._collection = _agt(connection, 'collection', str,
                                "Please provide a valid string for mongodb collection.")

        self._img_location = _agt(
            location, 'image', str, "Please provide a valid location for the image binary field in the selected mongodb collection.")
        self._lbl_location = _agt(
            location, 'label', str, "Please provide a valid location for the label field in the selected mongodb collection.")

        self._batchsize = _agt(
            config, 'batchsize', int, "Please select a valid integer value for the batchsize parameter.")
        self._shuffle = _agt(
            config, 'shuffle', bool, "Please select a valid boolean value for the shuffle parameter.")
        self._seed = _agt(
            config, 'seed', int, "Please select a valid integer value for the seed parameter.")
        self._height = _agt(
            config, 'height', int, "Please select a valid integer value for the image height parameter.")
        self._width = _agt(
            config, 'width', int, "Please select a valid integer value for the image width parameter.")

        self._dtype = K.floatx()
        self._size = (self._height, self._width)
        self._object_ids = self.__getOBIDS(query)

        self._samples = len(self._object_ids)
        assert (self._samples > 0), "The resulted query returned zero(0) samples."
        assert (self._samples >
                self._batchsize), "The resulted query returned less samples than the selected batchsize."

        self._dictionary, self._classes = self.__getDictionary()
        print(self._dictionary)
        assert (self._classes >
                1), "The resulted query return insufficient distinct classes."

        super(MongoGenerator, self).__init__(self._samples,
                                             self._batchsize, self._shuffle, self._seed)

    def __getOBIDS(self, query):
        collection = self.__connect()
        object_ids = collection.distinct("_id", query)
        self.__disconnect(collection)
        return object_ids

    def __getDictionary(self):
        collection = self.__connect()
        lbls = collection.distinct(
            self._lbl_location, {'_id': {'$in': self._object_ids}})
        nb = len(lbls)
        # keys as human readable, any type.
        dictionary = {k: self.__hot(v, nb) for v, k in enumerate(lbls)}
        self.__disconnect(collection)
        return dictionary, nb

    def __hot(self, idx, nb):
        hot = np.zeros((nb,))
        hot[idx] = 1
        return hot

    def _get_batches_of_transformed_samples(self, index_array):

        batch_x = np.zeros((len(index_array), self._height,
                            self._width, 3), dtype=self._dtype)

        batch_y = np.zeros(
            (len(index_array), self._classes), dtype=self._dtype)

        for i, j in enumerate(index_array):

            # Get sample data
            (x, y) = self.__readMongoSample(self._object_ids[j])

            self.image_data_generator.fit(x)

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
        sample = collection.find_one({'_id': oid})
        assert sample != None, "Failed to retrieve the sample corresponding to the image ID: " + \
            str(oid)
        return (self.__getImage(sample), self.__getLabel(sample))

    def __connect(self):
        return pymongo.MongoClient(self._host, self._port)[self._database][self._collection]

    def __disconnect(self, collection):
        del collection
        collection = None

    def __getImage(self, sample):
        strg = _ag(sample, self._img_location, "Failed to retrieve image binary (ID:" +
                   str(_get(sample, '_id'))+") at "+self._img_location+".")
        img = Image.open(io.BytesIO(strg)).resize(self._size).convert("RGB")
        return np.asarray(img, dtype=self._dtype)[..., :3]

    def __getLabel(self, sample):
        idstr = str(_get(sample, '_id'))
        label = _ag(sample, self._lbl_location,
                    "Failed to retrieve image label (ID:"+idstr+") at "+self._lbl_location+".")
        return self.getEncoded(label)

    def getShape(self):
        return (self._height, self._width, 3)

    def getClassNumber(self):
        return self._classes

    def getEncoded(self, label):
        return _get(self._dictionary, label)

    def getDecoded(self, np):
        return self._dictionary.keys()[self._dictionary.values().index(np)]


class MongoImageDataGenerator(ImageDataGenerator):
    self.total_samples = 0
    if(self.zca_whitening):
        assert (scipy is not None),'Using zca_whitening requires SciPy.Please install SciPy'

    def fit(self, x,
            augment=True,
            rounds=1,
            seed=None):
        """Fits the data generator to the batch data on the fly. This function is automatically called in MongoIterator but can be also called by the user

        This computes the internal data stats related to the
        data-dependent transformations, based on an array of sample data.
        Our decision to run this all the times in order to keep up track of the generator used in Mokera

        # Arguments
            x: Batch data. Should have rank 4.
             In case of grayscale data,
             the channels axis should have value 1, in case
             of RGB data, it should have value 3, and in case
             of RGBA data, it should have value 4.
            augment: Boolean (default: False).
                Whether to fit on randomly augmented samples.
            rounds: Int (default: 1).
                If using data augmentation (`augment=True`),
                this is how many augmentation passes over the data to use.
            seed: Int (default: None). Random seed.
       """
        assert x.ndim == 4, 'Input to `.fit()` should have rank 4. Got array with shape: ' + \
            str(x.shape)
        assert isinstance(
            augment, bool), 'Augment parameter must be a valid boolean'
        assert (isinstance(rounds, int) & rounds >
                0), 'Rounds parameter must be a valid integer (at least 1)'

        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
                'Expected input to be images (as Numpy array) '
                'following the data format convention "' +
                self.data_format + '" (channels on axis ' +
                str(self.channel_axis) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' +
                str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' +
                str(x.shape) + ' (' + str(x.shape[self.channel_axis]) +
                ' channels).')

        if seed is not None:
            assert (isinstance(seed, int) & seed >
                    0), 'Seed parameter, if set, must be a valid intger (at least 1)'
            np.random.seed(seed)

        self.broadcast = [1, 1, 1]
        self.broadcast[self.channel_axis - 1] = x.shape[self.channel_axis]

        x = np.copy(x)
        if augment:
            ax = np.zeros(
                tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
                dtype=backend.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        self.batch_samples = np.shape(x)[0]

        if self.mean == None:
            self.mean = getMean(x)
        else:
            self.mean = (self.total_samples*self.mean) + (self.batch_samples *
                                                          getMean(x)) / (self.total_samples + self.batch_samples)

        x -= self.mean

        if self.std == None:
            self.std = getStd(x)
        else:
            self.std = (self.total_samples*self.std) + (self.batch_samples *
                                                        getStd(x)) / (self.total_samples + self.batch_samples)

        x /= (self.std + backend.epsilon())

        if self.principal_components == None:
            self.principal_components = getPrincipalComponents()
        else:
            pass

        self.total_samples += self.batch_samples

    def getMean(self, x):
        mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
        return np.reshape(mean, self.broadcast)

    def getStd(self, x):
        std = np.std(x, axis=(0, self.row_axis, self.col_axis))
        return np.reshape(std, self.broadcast)

    def getPrincipalComponents(self):
        flat_x = np.reshape(
            x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
        u, s, _ = scipy.linalg.svd(sigma)
        s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
        self.principal_components = (u * s_inv).dot(u.T)
