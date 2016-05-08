import theano
import theano.tensor as T
import numpy

rand_num_gen = numpy.random

N = 400 # training sample size
feats = 784 # number of input variables

# generate a dataset: D = (input_values, target_class)
D = (rand_num_gen.randn(N, feats), rand_num_gen.randint(size=N, low=0, high=2))
training_steps = 10000

# declare theano's symbolic variables
x = T.matrix('x')
y = T.vector('y')

# initialize the weight vector w randomly
weights = theano.shared(rand_num_gen.randn(feats), name="w")

# Initialize the bias term
bias = theano.shared(0.0, name='b')

# the weight and bias variables are shared to 
# keep their values between training iterations

print ("Initial model:")
print (w.get_value())
print (b.get_value())

#construct Theano expression graph
probability_1 = 1 / (1 + (T.exp(T.dot(x, weights) - bias)))
prediction = probability_1 > 0.5
xent = -y * T.log(probability_1) - (1-y) * T.log(1-probability_1)
cost = xent.mean() + 0.01 * (w ** 2).sum()
gw, gb = T.grad(cost, [weights, bias])

#compile
train = theano.function(inputs=[x,y], outputs=[prediction, xent], updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))

predict = theano.function(inputs=[x], outputs=prediction)


















































