from keras.layers import Dense, Input
from keras.models import clone_model, Model,Sequential

class Modify:
    def __new__(self,model,inp,out):
        ci,co = self.validation(model,inp,out)

        if(ci): #change input
            model = self.changeInp(model,inp)
        if(co): #change ouput
             model = self.changeOut(model,out)

        return model, any([ci,co])


    def validation(model,inp,out):
        assert isinstance(model,Sequential) or isinstance(model,Model)
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
        
    def changeInp(model,inp):
        return clone_model(model,Input(batch_shape=inp))
        
    def changeOut(model,out):
        idx = self.findPreTop(model) # Finds the pre-topping layer (must be tested more extensively)
        preds = self.reshapeOutput(model,idx,out)
        model = Model(inputs=model.input, outputs=preds)
  
    def findPreTop(model):
        i = len(model.layers)-1
        cos = model.output_shape
        while(model.layers[i].output_shape == cos):
            i -= 1
        return i
 
    def reshapeOutput(model,i,out): # Reshapes model accordingly to https://keras.io/applications/#usage-examples-for-image-classification-models
        layer=model.layers[i]
        x = layer.output
        x = Dense(out[1], activation='softmax')(x)
        return x