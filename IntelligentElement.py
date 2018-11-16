from keras import Model
import keras.layers as L
import numpy as np

def recursiveFill(padded_input, vec, cur_slice):
    #print(cur_slice)
    ss = cur_slice
    if isinstance(vec, np.ndarray):
        for p in vec.shape:
            ss += (slice(0,p),)        
        padded_input[ss] = vec
    elif isinstance(vec, list):
        cur_n=0
        for v in vec:
            ss=cur_slice + ((cur_n,))
            recursiveFill(padded_input, v, ss)
            cur_n+=1
    else:
        raise ValueError('Error filling array: expected data to eventually become a numpy array')

def get_children_sum_last_output_shapes(children_ie):
    '''
    Retrieves the sum of the last dimension of shape of children

    This is usually number of dimensions of RNN or number of channels of images,
    useful to build model
    '''
    if children_ie is None:
        return 0
    if not isinstance(children_ie, list):
        children_ie=[children_ie]
    return sum([x.model.output_shape[-1] for x in children_ie])

def getMaxShape(x):
    #print(len(x))
    if isinstance(x,np.ndarray):
        return x.shape
    elif isinstance(x, list):
        lstshapes = [getMaxShape(e) for e in x]
        maxshape = np.max(lstshapes, axis=0)
        if maxshape[0]>0:
            maxshape = np.concatenate([np.array([len(x)]), maxshape])
        else:
            maxshape = np.array([len(x)])
        return maxshape
    return np.array([0])

def identity_fcn(x):
    return x

class IntelligentElement:
    '''
    Implements IntelligentElement, a base class that can be used to automate Keras model creation
    for complex nested structures
    '''
    
    def __init__(self, data, model, input_shape, preprocess_function = None, children_ie = None, name=None):
        '''
        Initializes the IntelligentElement instance with its data, which must be a list.
        
        All data passed to children IntelligentElements should be a list with the same number of elements
        and matching information. If a dynamic axis is found (input_shape[k] = None for some k), the vectors are 
        padded with zeroes until they get to the max length
        
        data        - list of samples that will be handled by this IE. If there is no model, it can contain empty elements
        children_ie - IEs that handle nested structures
        
        preprocess_function - function that is applied to each element of data list in order 
                              to retrieve neural network ready data as numpy arrays
        
        model - None if there is no data associated, otherwise a Keras Model
        '''
        
        assert isinstance(data, list), 'data should be a list of samples to be handled by this IntelligentElement'
        
        if children_ie is not None:
            if not isinstance(children_ie, list):
                children_ie=[children_ie]

            for c in children_ie:
                assert isinstance(c, IntelligentElement), 'children_ie must contain only IntelligentElement'
                assert len(c.data) == len(data), 'length of data vector must be the same for parents and children'
        
        if model is not None:
            assert isinstance(model, Model), 'model should be a Keras model'
            #the shape requirement is that model should handle a concatenation of parent input and children output
            
            expected_input_shape = (*model.input_shape[:-1], get_children_sum_last_output_shapes(children_ie) + input_shape[-1])
            assert model.input_shape[-1] == expected_input_shape[-1], 'Model should handle a concatenation of its input and children outputs. Expected model.input_shape={}, got {}'.format( expected_input_shape, model.input_shape)
            
            
        if preprocess_function is None:
            self.preprocess_function = identity_fcn
        else:
            self.preprocess_function = preprocess_function
            
            
        self.name=name
        self.data = data
        self.input_shape = input_shape
        self.children_ie = children_ie
        self.model = model
        
        if model is not None:
            self.model.name = 'm_{}'.format(name)

    def retrieve_model_inputs_outputs(self):
        inps=[]
        outs=[]
        
        if self.children_ie is not None:
            for c in self.children_ie:
                cmodel, cinp, cout = c.retrieve_model_inputs_outputs()
                inps += cinp
                outs.append(cout)
                
        if self.model is not None:
            inp = L.Input(self.input_shape, name='inp_{}'.format(self.name))
            inps.append(inp)
            outs.append(inp)
            
            #print([q.shape for q in outs])
            if len(outs) > 1:
                o = L.Concatenate()(outs)
            else:
                o = outs[0]
            #print(o.shape)
            outs = [self.model(o)]
            
        if len(outs) > 1:
            o = L.Concatenate()(outs)
        else:
            o = outs[0]
        
        ret_model = Model(inputs = inps, outputs = o, name = self.name)
        
        return ret_model, inps, o
    
    
    def get_batch(self, indices):
        '''
        Retrieves a batch of data as requested in indices
        '''
        inps=[]
        batch_data = [self.data[i] for i in indices]

        if self.children_ie is not None:
            for c in self.children_ie:
                cinp = c.get_batch(indices)
                inps += cinp
        
        if self.model is not None:
            cur_inps = [self.preprocess_function(x) for x in batch_data]
            
            #if there are no dynamic axes, we are done. if not, we need to pad
            shapes = np.array([getMaxShape(x) for x in cur_inps])
            maxshape = np.max(shapes,axis=0)
            
            #print(maxshape)
            padded_inp = np.zeros( (len(indices),*maxshape) )
            
            #print('Padded shape: {}'.format(padded_inp.shape))
            
            
            recursiveFill(padded_inp, cur_inps, ())
            
            #cur_n=0
            #for vec in cur_inps:
            #    ss = ( (cur_n,) )
            #    for p in vec.shape:
            #        ss += (slice(0,p),)
                    
                #print(ss)
                #print(np.array(vec))
                #print(padded_inp.shape)
                
            #    padded_inp[ss] = vec
            #    cur_n+=1
            
            inps.append(padded_inp)
            
            
        return inps