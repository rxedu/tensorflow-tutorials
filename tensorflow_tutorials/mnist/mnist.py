"""
[MNIST For ML Beginners][MNIST]

[MNIST]: https://www.tensorflow.org/versions/r0.8/tutorials/mnist/beginners/index.html

    from tensorflow_tutorials.mnist import MNIST
    from tensorflow.examples.tutorials.mnist import input_data

    data = input_data.read_data_sets('data', one_hot=True)

    mnist = MNIST()
    session = mnist.train(data.train)
    accuracy = mnist.check_accuracy(data.test, session)
    print(accuracy) # => 0.9203
    session.close()
"""
import tensorflow as tf

class MNIST:
    """Model for MNIST."""
    PIXELS = 784
    DIGITS = 10

    def __init__(self, learning_rate=0.5, training=1000, batches=100):
        self._inputs = None
        self._train_step = None
        self._model = None
        self._distribution = None
        self._entropy = None
        self._init = None
        self._config = {
            'learning_rate': learning_rate,
            'training': training,
            'batches': batches
        }
        self.init()

    @property
    def inputs(self):
        """The inputs variable."""
        if self._inputs is not None:
            return self._inputs

        self._inputs = tf.placeholder(tf.float32, [None, self.PIXELS])

        return self._inputs

    @property
    def model(self):
        """The softmax model."""
        if self._model is not None:
            return self._model

        weights = tf.Variable(tf.zeros([self.PIXELS, self.DIGITS]))
        bias = tf.Variable(tf.zeros([self.DIGITS]))

        self._model = tf.nn.softmax(tf.matmul(self.inputs, weights) + bias)

        return self._model

    @property
    def distribution(self):
        """The true distribution."""
        if self._distribution is not None:
            return self._distribution

        self._distribution = tf.placeholder(tf.float32, [None, 10])

        return self._distribution

    @property
    def entropy(self):
        """The cross entropy between the model and true distribution."""
        if self._entropy is not None:
            return self._entropy

        self._entropy = (
            tf.reduce_mean(-tf.reduce_sum(
                self.distribution * tf.log(self.model),
                reduction_indices=[1]))
        )

        return self._entropy

    @property
    def train_step(self):
        """
        The training step which minimizes the entropy
        using the gradient decent algorithm.
        """
        if self._train_step is not None:
            return self._train_step

        self._train_step = (
            (tf.train
             .GradientDescentOptimizer(self._config['learning_rate'])
             .minimize(self.entropy))
        )

        return self._train_step

    def _get_vars(self):
        return [
            self.inputs,
            self.model,
            self.distribution,
            self.entropy,
            self.train_step
        ]

    def init(self):
        """Initialize all variables and return the init object."""
        if self._init is not None:
            return self._init

        self._get_vars()
        self._init = tf.initialize_all_variables()

        return self._init

    def new_session(self):
        """Return a new TensorFlow session."""
        session = tf.Session()
        session.run(self.init())
        return session

    def train(self, train_data, session=None):
        """
        Train the model with the data and return the session.
        Will create a new session if not given.
        """
        if session is None:
            session = self.new_session()

        for _ in range(self._config['training']):
            batch_xs, batch_ys = train_data.next_batch(self._config['batches'])

            session.run(self.train_step, feed_dict={
                self.inputs: batch_xs,
                self.distribution: batch_ys
            })

        return session

    def check_accuracy(self, test_data, session):
        """Return the accuracy of the model compared with the data."""
        correct_prediction = tf.equal(
            tf.argmax(self.model, 1),
            tf.argmax(self.distribution, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return session.run(accuracy, feed_dict={
            self.inputs: test_data.images,
            self.distribution: test_data.labels
        })
