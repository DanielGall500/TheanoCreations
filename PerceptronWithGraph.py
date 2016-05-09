import theano
import theano.tensor as T
import numpy as np

ninputs = 1000 #perceptron inputs
nfeatures = 100 #perceptron features
noutputs = 10 #perceptron outputs
nhiddens = 50 #hidden layers

rand_num_gen = np.random.RandomState(42)
x = T.matrix('x')

weightH = theano.shared(rand_num_gen.normal(0, 1, (nfeatures, nhiddens)), borrow=True)
biasH = theano.shared(np.zeros(nhiddens), borrow=True)

hidden = T.nnet.sigmoid(T.dot(x, weightH) + biasH)

weightY = theano.shared(rand_num_gen.normal(0, 1, (nhiddens, noutputs)))
biasY = theano.shared(np.zeros(noutputs), borrow=True)

y = T.nnet.softmax(T.dot(hidden, weightY) + biasY)

predict = theano.function(inputs=[x], outputs=y) #outputs the probability of 10 classes

import pydot
import theano.d3viz as d3v
import os

if not os.path.exists('/Users/dannyg/Desktop/Projects/TheanoCreations/Outputs'):
	os.makedirs('/Users/dannyg/Desktop/Projects/TheanoCreations/Outputs')

d3v.d3viz(predict, '/Users/dannyg/Desktop/Projects/TheanoCreations/Outputs/graph.html')












