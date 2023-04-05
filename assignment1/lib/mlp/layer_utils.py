from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v
    
    def apply_l1_regularization(self, lam):
        """
        Gather gradients for L1 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ######## TODO ########
                param = self.params[n]
                grad = (param > 0).astype(np.float32) - (param < 0).astype(np.float32)
                self.grads[n] += lam * grad
                ######## END  ########
    
    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ######## TODO ########
                self.grads[n] += lam * self.params[n]
                ######## END  ########


    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))


class flatten(object):
    def __init__(self, name="flatten"):
        """
        - name: the name of current layer
        - meta: to store the forward pass activations for computing backpropagation
        Note: params and grads should be just empty dicts here, do not update them
        """
        self.name = name
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, feat):
        #print(output.shape)
        #############################################################################
        # TODO: Implement the forward pass of a flatten layer.                      #
        # You need to reshape (flatten) the input features.                         #
        # Store the results in the variable self.meta provided above.               #
        #############################################################################
        output = feat.reshape(feat.shape[0], np.prod(feat.shape[1:]))
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        #############################################################################
        # TODO: Implement the backward pass of a flatten layer.                     #
        # You need to reshape (flatten) the input gradients and return.             #
        # Store the results in the variable dfeat provided above.                   #
        #############################################################################
        pass
        dfeat = dprev.reshape(feat.shape)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dfeat


class fc(object):
    def __init__(self, input_dim, output_dim, init_scale=0.002, name="fc"):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - input_dim: input dimension
        - output_dim: output dimension
        - meta: to store the forward pass activations for computing backpropagation
        """
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(input_dim, output_dim)
        self.params[self.b_name] = np.zeros(output_dim)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None

    def forward(self, feat):
        assert len(feat.shape) == 2 and feat.shape[-1] == self.input_dim, \
            "But got {} and {}".format(feat.shape, self.input_dim)
        #############################################################################
        # TODO: Implement the forward pass of a single fully connected layer.       #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        pass
        bias = np.reshape(self.params[self.b_name],(self.params[self.b_name].shape[0],1))
        output = np.dot(self.params[self.w_name].T,feat.T) + bias
        output = output.T
        #print(output.shape)
        #print(self.params[self.b_name].shape)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        #############################################################################
        # TODO: Implement the backward pass of a single fully connected layer.      #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        pass
        #print(dprev.shape)
        dfeat = np.dot(dprev, self.params[self.w_name].T)
        self.grads[self.w_name] = np.dot(feat.T, dprev)
        self.grads[self.b_name] = np.reshape(np.dot(np.ones((1, dprev.shape[0])), dprev), (self.output_dim,))
        #print(dprev.shape, self.output_dim)
        assert len(feat.shape) == 2 and feat.shape[-1] == self.input_dim, \
            "But got {} and {}".format(feat.shape, self.input_dim)
        assert len(dprev.shape) == 2 and dprev.shape[-1] == self.output_dim, \
            "But got {} and {}".format(dprev.shape, self.output_dim)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dfeat

class gelu(object):
    def __init__(self, name="gelu"):
        """
        - name: the name of current layer
        - meta:  to store the forward pass activations for computing backpropagation
        Notes: params and grads should be just empty dicts here, do not update them
        """
        self.name = name 
        self.params = {}
        self.grads = {}
        self.meta = None 
    
    def forward(self, feat):
        output = 0.5  * np.multiply(feat, 1 + np.tanh(((2/np.pi)**(1/2)) * np.add(feat,(0.044715 * np.power(feat, 3)))))
        #############################################################################
        # TODO: Implement the forward pass of GeLU                                  #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output
    
    def backward(self, dprev):
        """ You can use the approximate gradient for GeLU activations """
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")                                                                           
        #############################################################################
        # TODO: Implement the backward pass of GeLU                                 #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        pass
        sech = 1 / np.power(np.cosh(math.sqrt(2 / math.pi) * np.add(feat, 0.044715*np.power(feat,3))),2)
        d = 1 + (0.134145 * np.power(feat, 2))
        gradient = 0.5 * ((math.sqrt(2/ math.pi) * feat * sech * d) + np.tanh(math.sqrt(2 / math.pi) * np.add(feat, 0.044715*np.power(feat,3))) + 1)
        dfeat = gradient * dprev
        # dfeat = None
        # tan_h = np.tanh((np.add(0.0356774 * np.power(feat, 3), 0.797885 * feat)))
        # sec_h = np.power(np.cosh((np.add(0.0356774 * np.power(feat, 3), 0.797885 * feat))), -2)
        # eq = np.add(0.0535161 * np.power(feat, 3), 0.398942 * feat)
        # grad = 0.5 + np.add((0.5 * tan_h), np.multiply(eq, sec_h))
        # dfeat = np.multiply(grad, dprev)         
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        #print(dfeat.shape)
        return dfeat



class dropout(object):
    def __init__(self, keep_prob, seed=None, name="dropout"):
        """
        - name: the name of current layer
        - keep_prob: probability that each element is kept.
        - meta: to store the forward pass activations for computing backpropagation
        - kept: the mask for dropping out the neurons
        - is_training: dropout behaves differently during training and testing, use
                       this to indicate which phase is the current one
        - rng: numpy random number generator using the given seed
        Note: params and grads should be just empty dicts here, do not update them
        """
        self.name = name
        self.params = {}
        self.grads = {}
        self.keep_prob = keep_prob
        self.meta = None
        self.kept = None
        self.is_training = False
        self.rng = np.random.RandomState(seed)
        assert keep_prob >= 0 and keep_prob <= 1, "Keep Prob = {} is not within [0, 1]".format(keep_prob)

    def forward(self, feat, is_training=True, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        #############################################################################
        # TODO: Implement the forward pass of Dropout.                              #
        # Remember if the keep_prob = 0, there is no dropout.                       #
        # Use self.rng to generate random numbers.                                  #
        # During training, need to scale values with (1 / keep_prob).               #
        # Store the mask in the variable kept provided above.                       #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        pass
        D = self.rng.random(feat.shape)
        if is_training == True:
            D = D < self.keep_prob
            if self.keep_prob == 0:
                output = feat
            else:
                output = 1/self.keep_prob * np.multiply(feat, D) 
        else:
            output = feat
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.kept = D
        self.is_training = is_training
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        #############################################################################
        # TODO: Implement the backward pass of Dropout                              #
        # Select gradients only from selected activations.                          #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        pass
        if self.is_training == True and self.keep_prob != 0:
            dfeat = np.multiply(dprev,self.kept) * (1/self.keep_prob)
        else: 
            dfeat = dprev
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.is_training = False
        self.meta = None
        return dfeat


class cross_entropy(object):
    def __init__(self, size_average=True):
        """
        - size_average: if dividing by the batch size or not
        - logit: intermediate variables to store the scores
        - label: Ground truth label for classification task
        """
        self.size_average = size_average
        self.logit = None
        self.label = None

    def forward(self, feat, label):
        logit = softmax(feat)
        loss = None
        #############################################################################
        # TODO: Implement the forward pass of an CE Loss                            #
        # Store the loss in the variable loss provided above.                       #
        #############################################################################
        pass
        #log_likelihood = -np.log(logit[self.size_average,label])
        #loss = np.sum(log_likelihood) / self.size_average
        #if size_average =
        #loss = - np.sum(np.log(logit) * label[:, None])/self.size_average
        b = np.zeros((feat.shape))
        b[np.arange(label.size), label] = 1
        #print(logit.shape, max(label))
        label = b
        loss = -(np.sum(np.multiply(label, np.log(logit))))/label.shape[0]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.logit = logit
        self.label = label
        return loss

    def backward(self):
        logit = self.logit
        label = self.label
        if logit is None:
            raise ValueError("No forward function called before for this module!")
        dlogit = None
        #############################################################################
        # TODO: Implement the backward pass of an CE Loss                           #
        # Store the output gradients in the variable dlogit provided above.         #
        #############################################################################
        pass
        dlogit = (logit - label)/label.shape[0]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.logit = None
        self.label = None
        return dlogit


def softmax(feat):
    scores = None
    #############################################################################
    # TODO: Implement the forward pass of a softmax function                    #
    # Return softmax values over the last dimension of feat.                    #
    #############################################################################
    pass
    ex = np.exp(feat - np.max(feat, axis = 1, keepdims = True))
    scores = ex / np.sum((ex), axis = 1, keepdims = True)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return scores

def reset_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
