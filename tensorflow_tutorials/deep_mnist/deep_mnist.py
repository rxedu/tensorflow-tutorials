"""
[Deep MNIST for Experts][MNIST]

[MNIST]: https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html

    from tensorflow_tutorials.deep_mnist import DeepMNIST
    from tensorflow.examples.tutorials.mnist import input_data

    data = input_data.read_data_sets('data', one_hot=True)

    mnist = DeepMNIST(training=1000)
    session = mnist.train(data.train, test_data=data.test)
    accuracy = mnist.check_accuracy(data.test, session)
    print(accuracy) # => 0.9631
    session.close()
"""
from functools import partial, reduce

import tensorflow as tf

class DeepMNIST:
    """Neural net for MNIST."""
    PIXELS = 784
    DIGITS = 10
    FEATURES = 32
    SHAPE = [-1, 28, 28, 1]
    STDDEV = 0.1
    BIAS = 0.1
    CONV2D_STRIDES = [1, 1, 1, 1]
    POOL_KSIZE = [1, 2, 2, 1]
    POOL_STRIDES = [1, 2, 2, 1]
    KEEP_PROB = 0.5

    def __init__(self, neurons=1024,
                 learning_rate=1e-4, training=20000, batches=50):
        self._inputs = None
        self._inputs_image = None
        self._keep_prob = None
        self._train_step = None
        self._accuracy = None
        self._layers = None
        self._model = None
        self._distribution = None
        self._entropy = None
        self._init = None
        self._config = {
            'neurons': neurons,
            'learning_rate': learning_rate,
            'training': training,
            'batches': batches
        }

    @property
    def inputs(self):
        """The inputs variable."""
        if self._inputs is not None:
            return self._inputs

        self._inputs = tf.placeholder(tf.float32, [None, self.PIXELS])

        return self._inputs

    @property
    def inputs_image(self):
        """The reshaped inputs image variable."""
        if self._inputs_image is not None:
            return self._inputs_image

        self._inputs_image = tf.reshape(self.inputs, self.SHAPE)

        return self._inputs_image

    @property
    def keep_prob(self):
        """The keep probability."""
        if self._keep_prob is not None:
            return self._keep_prob

        self._keep_prob = tf.placeholder(tf.float32)

        return self._keep_prob

    @property
    def layers(self):
        """List of layer functions."""
        if self._layers is not None:
            return self._layers

        neurons = self._config['neurons']

        self._layers = [
            partial(self.convolutional_layer, [5, 5, 1, 32], [32]),
            partial(self.convolutional_layer, [5, 5, 32, 64], [64]),
            partial(self.connected_layer,
                    [7 * 7 * 64, neurons], [neurons], [-1, 7 * 7 * 64]),
            partial(self.dropout_layer, self.keep_prob),
            partial(self.readout_layer, [neurons, self.DIGITS], [self.DIGITS])
        ]

        return self._layers

    @property
    def model(self):
        """The model."""
        if self._model is not None:
            return self._model

        self._model = reduce(lambda x, f: f(x), self.layers, self.inputs_image)

        return self._model

    @property
    def distribution(self):
        """The true distribution."""
        if self._distribution is not None:
            return self._distribution

        self._distribution = tf.placeholder(tf.float32, [None, self.DIGITS])

        return self._distribution

    @property
    def entropy(self):
        """The cross entropy between the model and true distribution."""
        if self._entropy is not None:
            return self._entropy

        self._entropy = (
            tf.reduce_mean(-tf.reduce_sum(
                self.distribution * tf.log(self.model),
                axis=[1]))
        )

        return self._entropy

    @property
    def train_step(self):
        """
        The training step which minimizes the entropy
        using the Adam algorithm.
        """
        if self._train_step is not None:
            return self._train_step

        self._train_step = (
            (tf.train
             .AdamOptimizer(self._config['learning_rate'])
             .minimize(self.entropy))
        )

        return self._train_step

    @property
    def accuracy(self):
        """Training accuracy."""
        if self._accuracy is not None:
            return self._accuracy

        correct_prediction = tf.equal(
            tf.argmax(self.model, 1),
            tf.argmax(self.distribution, 1))

        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return self._accuracy

    def _get_vars(self):
        return [
            self.inputs,
            self.inputs_image,
            self.keep_prob,
            self.model,
            self.distribution,
            self.entropy,
            self.train_step,
            self.accuracy
        ]

    def init(self):
        """Initialize all variables and return the init object."""
        if self._init is not None:
            return self._init

        self._get_vars()
        self._init = tf.global_variables_initializer()

        return self._init

    def new_session(self):
        """Return a new TensorFlow session."""
        session = tf.Session()
        session.run(self.init())
        return session

    def train(self, train_data, test_data=None, session=None):
        """
        Train the model with the data and return the session.
        Will create a new session if not given.

        If `test_data` is given, will print out training accuracy
        every 100 training steps.
        """
        if session is None:
            session = self.new_session()

        for i in range(self._config['training']):
            if i % 100 == 0 and test_data:
                accuracy = session.run(self.accuracy, feed_dict={
                    self.inputs: test_data.images,
                    self.distribution: test_data.labels,
                    self.keep_prob: 1.0
                })
                print("step %d, training accuracy %g" % (i, accuracy))

            batch_xs, batch_ys = train_data.next_batch(self._config['batches'])

            session.run(self.train_step, feed_dict={
                self.inputs: batch_xs,
                self.distribution: batch_ys,
                self.keep_prob: self.KEEP_PROB
            })

        return session

    def check_accuracy(self, test_data, session):
        """Return the accuracy of the model compared with the data."""
        return session.run(self.accuracy, feed_dict={
            self.inputs: test_data.images,
            self.distribution: test_data.labels,
            self.keep_prob: 1.0
        })

    @classmethod
    def weight_variable(cls, shape):
        """Return a new weight variable."""
        initial = tf.truncated_normal(shape, stddev=cls.STDDEV)
        return tf.Variable(initial)

    @classmethod
    def bias_variable(cls, shape):
        """Return a new bias variable."""
        initial = tf.constant(cls.BIAS, shape=shape)
        return tf.Variable(initial)

    @classmethod
    def conv2d(cls, inputs, weights):
        """Return a new conv2d."""
        return tf.nn.conv2d(inputs, weights,
                            strides=cls.CONV2D_STRIDES,
                            padding='SAME')

    @classmethod
    def max_pool_2x2(cls, value):
        """Return a new max_pool."""
        return tf.nn.max_pool(value,
                              ksize=cls.POOL_KSIZE,
                              strides=cls.POOL_STRIDES,
                              padding='SAME')

    @classmethod
    def convolutional_layer(cls, weight_shape, bias_shape, inputs):
        """Return a new convolutional layer."""
        weights = cls.weight_variable(weight_shape)
        bias = cls.bias_variable(bias_shape)

        convolution = tf.nn.relu(cls.conv2d(inputs, weights) + bias)

        return cls.max_pool_2x2(convolution)

    @classmethod
    def connected_layer(cls, weight_shape, bias_shape, pool_shape, inputs):
        """Return a new connected layer."""
        weights = cls.weight_variable(weight_shape)
        bias = cls.bias_variable(bias_shape)
        pool = tf.reshape(inputs, pool_shape)

        return tf.nn.relu(tf.matmul(pool, weights) + bias)

    @classmethod
    def dropout_layer(cls, keep_prob, inputs):
        """Return a new dropout layer."""
        return tf.nn.dropout(inputs, keep_prob)

    @classmethod
    def readout_layer(cls, weight_shape, bias_shape, inputs):
        """Return a new readout layer."""
        weights = cls.weight_variable(weight_shape)
        bias = cls.bias_variable(bias_shape)

        return tf.nn.softmax(tf.matmul(inputs, weights) + bias)
