from keras.layers import Dense, Input
from keras.models import clone_model, Model, Sequential, model_from_json
from keras.callbacks import Callback
from keras import backend
from pymongo import MongoClient
from bson.objectid import ObjectId
import pydash as p_
import base64
import time
import tempfile
import gridfs
import io
import os
import pickle
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.models import Range1d
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from PIL import Image
import numpy as np


def _get(obj,loc):
    return p_.get(obj,loc)

def _getSafe(obj,loc,typ,msg):
    got = _get(obj,loc)
    assert isinstance(got,typ), msg
    return got

def connect(obj):
    return MongoClient(
        _getSafe(obj, 'host', str, 'Hostname is not a valid string.'),
        _getSafe(obj, 'port', int, 'Port is not a valid integer.'))[_getSafe(obj, 'database', str, 'Database name is not a valid string.')][_getSafe(obj, 'collection', str, 'Collection name is not a valid string.')]

def disconnect(collection):
    del collection
    collection = None

def ValidateModelArchitecture(model,inp,out):

    def validation(model,inp,out):
        assert isinstance(model,int) or isinstance(model,Model)
        n_in = len(model.inputs)
        n_out =len(model.outputs) 
        assert (n_in) > 0, 'Model has not detectable inputs.'
        assert (n_out) > 0, 'Model has not detectable outputs.'
        assert (n_in) <= 1, 'Model has multiple %d inputs tensors. Cannot apply input transformation.' % (n_in)
        assert (n_out) <= 1, 'Model has multiple %d output tensors Cannot apply output transformation.' % (n_out)
    
        inp_old = model.input_shape
    
        assert len(inp_old) == 4, 'Model input tensor shape != 4: Not a valid image classification model (B x X x X x X).'
    
        assert isinstance(inp,tuple), 'Input parameter is not a valid tuple.'
        assert len(inp) == 4, 'Input parameter is not a valid 4-rank tensor shape.'

        out_old = model.output_shape
        assert len(out_old) == 2, 'Model output tensor shape !=2: Not a valid image classification model (B x C).'
        assert isinstance(out,tuple), 'Output parameter is not a valid tuple.'
        assert len(out) == 2, 'Output parameter is not a valid 2-rank tensor shape.'

        ci = any([inp[i] != inp_old[i] for i in range(0,len(inp))])
        co = any([out[i] != out_old[i] for i in range(0,len(out))])
      
        return ci,co

    def findPreTop(model):
        i = len(model.layers)-1
        cos = model.output_shape
        while(model.layers[i].output_shape == cos):
            i -= 1
        return i
 
    def reshapeOutput(model,i,out): # Reshapes model accordingly to https://keras.io/applications/#usage-examples-for-image-classification-models
        return Dense(out[1], activation='softmax',name = 'output_aimune_'+str(int(round(time.time() * 1000))))(model.layers[i].output)
     
    def changeInp(model,inp):
        return clone_model(model,Input(batch_shape=inp, name = 'input_aimune_'+str(int(round(time.time() * 1000)))))
        
    def changeOut(model,out):
        idx = findPreTop(model) # Finds the pre-topping layer (must be tested more extensively)
        preds = reshapeOutput(model,idx,out)
        return Model(inputs=model.input, outputs=preds)

    ci,co = validation(model,inp,out)
        
    if(ci): #change input
        model = changeInp(model,inp)

    if(co): #change ouput
        model = changeOut(model,out)

    model.build(model.input_shape)

    return model, any([ci,co])

def LoadModelArchitecture(query,location,connection={'host':'localhost','port':27017,'database':'database','collection':'collection'}):
    assert (type(query)==dict and bool(query)), 'Please provide a valid dictionary identifying which model architecture to retrieve. Ex. "{"_id":xxx}"'
    assert type(location) == str, 'Please provide a valid location of the model architecture.'
    assert type(connection) == dict, 'Please provide a valid connection dictionary.' 
    
    col = connect(connection)
    doc = col.find_one(query)
    disconnect(col)
    arch = _get(doc,location)

    if(isinstance(arch,str)):
        return _to_model(arch)
    elif(isinstance(arch,bytes)):
        arch2 = arch.decode()
        return _to_model(arch2)
    elif(arch is None):
        raise ValueError('Could not find an architecture object in the database.')
    else:
        raise TypeError('Detected architecture object is neither a string or binary json object.')

def _to_model(archi):
    assert type(archi)==str, 'Retrieved field (decoded) is not a valid Keras model json string.' 
    try:
            return model_from_json(archi) #simply encoded as binary or loaded as pure string
    except:           
            return model_from_json(base64.b64decode(archi).decode()) # the string was encoded in base64
class MonkeraCallback(Callback):
    def __init__(self,query,config={'ini':0,'end':1,'epochs':1,'value':'value','description':'description'},connection={'host':'localhost','port':27017,'database':'database','collection':'collection'}):
         super(MonkeraCallback, self).__init__()
         assert type(query) == dict and bool(query), 'Please insert a valid dictionary for the query parameter. Ex:"{"_id":xxx}"'
         self.query = query
         self.ini = _getSafe(config,'ini',(int,float),'Initialization parameter must be an integer or a float.')
         self.end = _getSafe(config,'end',(int,float),'Ending parameter must be an integer or a float.')
         assert self.end > self.ini ,'Ending parameter must be bigger than initialization parameter.'
         assert self.ini>=0 and self.ini<1, 'Initialization parameter must be between 0 inclusive and 1 exclusive.'
         assert self.end>0 and self.ini<=1, 'Ending parameter must be between 0 exclusive and 1 inclusive.'

         self.valuepath = _getSafe(config,'value',str, 'Value parameter must be a valid string referencing the path of the value to write in the document.')
         self.descriptionpath = _getSafe(config,'description',str, 'Description parameter must be a valid string referencing the path of the description to write in the document.')

         self.epochs = _getSafe(config,'epochs',int,'Epochs parameter must be a valid integer value.')
         assert self.epochs > 0, 'Epochs parameter must be positive.'

         _getSafe(connection,'host',str,'Host must be a valid string referencing to mongodb host.')
         _getSafe(connection,'port',int,'Port must be a valid integer referencing to mongodb port.')
         _getSafe(connection,'database',str,'Database must be a valid string referencing to mongodb database.')
         _getSafe(connection,'collection',str,'Collection must be a valid string referencing to mongodb collection.')
         self.connection = connection

    def on_epoch_end(self, epoch, logs = None):

            value = self.ini + (epoch/self.epochs)*(self.end-self.ini)
            description = "Fitting model: Epoch: "+str(epoch)+ " Loss: " + str(logs.get('loss'))
            col = connect(self.connection)
            col.update_one(self.query,{"$set":{self.valuepath:value,
                self.descriptionpath:description}})
            disconnect(col)

def SaveModelWeights(model,connection={'host':'localhost','port':27017,'database':'database'}):
    db = MongoClient(_getSafe(connection,'host',str,'Host must be a valid string referencing to mongodb host.'),
         _getSafe(connection,'port',int,'Port must be a valid integer referencing to mongodb port.'))[_getSafe(connection,'database',str,'Database must be a valid string referencing to mongodb database.')]
    try:
        fs = gridfs.GridFS(db)
        fd, name = tempfile.mkstemp()
        model.save_weights(name)
        return fs.put(open(fd, 'rb'))
    finally:
        os.remove(name)
        disconnect(db)

def LoadModelWeights(model,fileID,connection={'host':'localhost','port':27017,'database':'database'}):
    db = MongoClient(_getSafe(connection,'host',str,'Host must be a valid string referencing to mongodb host.'),
         _getSafe(connection,'port',int,'Port must be a valid integer referencing to mongodb port.'))[_getSafe(connection,'database',str,'Database must be a valid string referencing to mongodb database.')]
    assert isinstance(fileID,ObjectId) , 'Provided fileID parameter is not a valid ObjectID.'
    try:
        fs = gridfs.GridFS(db)

        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(fs.get(fileID).read())
        f.close()
        model.load_weights(f.name)
    finally:
        os.unlink(f.name)
        disconnect(db)

def SaveHistory(history,connection={'host':'localhost','port':27017,'database':'database'}):
        db = MongoClient(_getSafe(connection,'host',str,'Host must be a valid string referencing to mongodb host.'),
         _getSafe(connection,'port',int,'Port must be a valid integer referencing to mongodb port.'))[_getSafe(connection,'database',str,'Database must be a valid string referencing to mongodb database.')]        
        try:
            fs = gridfs.GridFS(db)
            f = writeTempPickle(history)
            return fs.put(open(f.name, 'rb'))
        finally:
            os.remove(f.name)
            disconnect(db)

def LoadHistory(fileID,connection={'host':'localhost','port':27017,'database':'database'}):
    db = MongoClient(_getSafe(connection,'host',str,'Host must be a valid string referencing to mongodb host.'),
         _getSafe(connection,'port',int,'Port must be a valid integer referencing to mongodb port.'))[_getSafe(connection,'database',str,'Database must be a valid string referencing to mongodb database.')]
    assert isinstance(fileID,ObjectId) , 'Provided fileID parameter is not a valid ObjectID.'
    try:
        fs = gridfs.GridFS(db)
        out = fs.get(fileID)
        return readTempPickle(out)
    finally:
        disconnect(db)

def writeTempPickle(obj):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        pickle.dump(obj, f)
    return f

def readTempPickle(out):
    return pickle.load(out)
   
def PlotHistory(history,width=300,height=300,tools="pan,wheel_zoom,box_zoom,reset,save"):
    assert isinstance(history,dict), 'Please provide a valid history dictionary parameter.' 
    assert isinstance(width,int) and width>0, 'Please provide a valid value for the width of the graphical component.'
    assert isinstance(height,int) and height>0, 'Please provide a valid value for the height of the graphical component.'
    assert isinstance(tools,str), 'Please insert a valid string value for the tools parameter'

    #Dictionary of unique plots keys
    unique_metrics = set(list([x.replace('val_','') for x in history.keys()]))

    plots=list()
    
    for x in unique_metrics:
        
        plot = figure(title=x,tools=tools,x_axis_label='Epochs', y_axis_label='Value',plot_width=width, plot_height=height)
        datax = _get(history,x)
        if(datax):
            rangem = len(datax)+1
            plot.line(range(1,rangem), datax, legend=x, color="blue")
            plot.circle(range(1,rangem), datax, legend=x, color="blue",fill_color="white", size=5)

        dataxval = _get(history,'val_'+x)
        if(dataxval):
            rangem = len(dataxval)+1
            plot.line(range(1,rangem), dataxval, legend='val_'+x, color="green")
            plot.circle(range(1,rangem), dataxval, legend='val_'+x, color="green",fill_color="white", size=5)
        plots.append(plot)

    return plots

def PlotPredictions(preds,labels,tools="pan,wheel_zoom,box_zoom,reset,save",height=300,width=300):

    assert isinstance(preds,np.ndarray), 'Please provide a valid preds numpy array as parameter.' 
    assert isinstance(labels,dict),'Please provide a valid labels'
    assert isinstance(width,int) and width>0, 'Please provide a valid value for the width of the graphical component.'
    assert isinstance(height,int) and height>0, 'Please provide a valid value for the height of the graphical component.'
    assert isinstance(tools,str), 'Please insert a valid string value for the tools parameter'

    source = ColumnDataSource(data=dict(legend=list(labels.keys()), counts=preds.tolist(), color=Spectral6))
    plot = figure(title='Results Histogram',tools=tools,x_axis_label='Classes', y_axis_label='%',plot_width=width, plot_height=height)
    plot.vbar(x='legend', top='counts', width=0.8, color='color', source=source)
    
    return plot

def ValidateImageModel(query,location,connection={'host':'localhost','port':27017,'database':'database','collection':'collection'},config={'target_size':(100,100),'color_format':'L','data_format':'channels_last'}):
    assert isinstance(query,dict), 'Query is not a valid dictionary.'
    assert isinstance(location,str), 'Location is not a valid string.'
    assert isinstance(connection,dict), 'Collection is not a valid dictionary.'
    assert isinstance(config,dict), 'Config is not a valid dictionary.'
    _getSafe(connection,'host',str,'Connection host is not a valid string.')
    _getSafe(connection,'port',int,'Connection port is not a valid integer.')
    _getSafe(connection,'database',str,'Connection database is not a valid string.')
    _getSafe(connection,'collection',str,'Connection collection is not a valid string.')

    target_size = _getSafe(config,'target_size',tuple,'Config target size is not a valid tuple.')
    assert len(target_size) == 2, 'Target_size tuple is not 2 dimensional.'

    data_format = _getSafe(config, 'data_format', str, "Please select a valid string for the data_format parameter.")
    assert (data_format in ['channels_first', 'channels_last']), 'Data format parameter should be `"channels_last"` (channel after row and column) or `"channels_first"` (channel before row and column). Received: %s' % self.data_format

    color_format = _getSafe(config, 'color_format', str,"Please select a valid string for color_format parameter.")
    assert color_format in ['L','RGB','RGBA'], "Please select a valid color format string: L, RGB, or RGBA"

    col = connect(connection)
    imgstr = _getSafe(col.find_one(query),location,(str,bytes),'Selected image object is not a valid str/binary image object.')
    disconnect(col)

    img = Image.open(io.BytesIO(imgstr)).resize(target_size).convert(color_format)
    tensor = img_to_array(img,data_format=data_format)

    return np.array([tensor,])

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=backend.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x