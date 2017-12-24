# FeedForwardModel
A library that enables incredibly **simple creation of feed forward neural models** in TensorFlow.

Simply specify the input shape, output shape, and a list of components (or layers) to make up the network.
The library then figures out the dimensions of each component, initializes each component, and builds the model automatically.
Most importantly, the model is able to infer the sizes of intermediate layers automatically, requiring you to only specify the essentials.

This enables you to make quick changes to your entire model in just a few lines.

## Examples

**fc_test.py**: An example implementation of a simple multi-class classifier.

The model used in this example:
~~~~
...
model = FeedForwardModel(2, 4, [FullyConnectedComponent(8),
                                ActivationComponent(tf.nn.sigmoid),
                                DropoutComponent(0.99),
                                FullyConnectedComponent()])
x, out = model.build()
...
~~~~

---
**cnn_test.py**: An example implementation of a CNN. Uses the MNIST dataset.

The model used in this example:
~~~~
...
model = FeedForwardModel([28, 28], 10, [Convolutional2DComponent(5, num_kernels=20),
                                        ActivationComponent(tf.nn.relu),
                                        DropoutComponent(0.95),
                                        FullyConnectedComponent()])
x, out = model.build()
...
~~~~
