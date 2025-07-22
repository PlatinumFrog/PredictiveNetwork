# Predictive Network
This is a Recurrent Predictive Coding neural network with unified IO.

# Predictive Coding
Predictive Coding is a model of deep learning that more closely resembles the learning mechanisms of the brain than the popular back-propagation. Rather than pausing the network and cascading derivatives backwards through the network, each neuron and weight is updated independently using only local information.

# Recurrent Unified IO
Rather than the network having layers, the network contains one big layer and weights connecting each node to every other node. Any node in the network can be associated with an input or output signal. The recurrance allows information to propagate anywhere in the network. This way, the network learns its own topology rather than being restricted to predefined layers.
