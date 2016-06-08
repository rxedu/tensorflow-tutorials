# pylint: disable=import-error
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name

import pytest

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_tutorials.deep_mnist import DeepMNIST

@pytest.fixture(scope='session')
def mnist_data(tmpdir_factory):
    data_dir = tmpdir_factory.mktemp('data').join('mnist')
    return input_data.read_data_sets(str(data_dir), one_hot=True)

def test_inputs():
    assert DeepMNIST().inputs is not None

def test_inputs_image():
    assert DeepMNIST().inputs_image is not None

def test_keep_prob():
    assert DeepMNIST().keep_prob is not None

def test_layers():
    assert DeepMNIST().layers is not None

def test_model():
    assert DeepMNIST().model is not None

def test_distribution():
    assert DeepMNIST().distribution is not None

def test_entropy():
    assert DeepMNIST().entropy is not None

def test_train_step():
    assert DeepMNIST().train_step is not None

def test_init():
    assert DeepMNIST().init is not None

def test_train(mnist_data):
    session = DeepMNIST(training=100).train(mnist_data.train,
                                            test_data=mnist_data.test)
    assert session is not None
    session.close()

def test_check_accuracy(mnist_data):
    mnist = DeepMNIST(training=500)
    session = mnist.train(mnist_data.train)
    accuracy = mnist.check_accuracy(mnist_data.test, session)
    assert accuracy >= 0.94
    assert accuracy <= 0.96
    session.close()

def test_weight_variable():
    assert DeepMNIST().weight_variable([1024, 10]) is not None

def test_bias_variable():
    assert DeepMNIST().bias_variable([10]) is not None

def test_conv2d():
    inputs = DeepMNIST().inputs_image
    weights = DeepMNIST().weight_variable([5, 5, 1, 32])
    assert DeepMNIST().conv2d(inputs, weights) is not None

def test_max_pool_2x2():
    inputs = DeepMNIST().inputs_image
    assert DeepMNIST().max_pool_2x2(inputs) is not None

def test_convolutional_layer():
    inputs = DeepMNIST().inputs_image
    assert DeepMNIST().convolutional_layer(
        [5, 5, 1, 32], [32], inputs) is not None

def test_connected_layer():
    inputs = DeepMNIST().inputs_image
    assert DeepMNIST().connected_layer(
        [7 * 7 * 64, 1024], [1024], [-1, 7 * 7 * 64], inputs) is not None

def test_dropout_layer():
    inputs = DeepMNIST().inputs_image
    keep_prob = DeepMNIST().keep_prob
    assert DeepMNIST().dropout_layer(keep_prob, inputs) is not None

def test_readout_layer():
    inputs = DeepMNIST().inputs_image
    layer_1 = DeepMNIST().convolutional_layer([5, 5, 1, 32], [32], inputs)
    layer_2 = DeepMNIST().connected_layer(
        [7 * 7 * 32, 1024], [1024], [-1, 7 * 7 * 32], layer_1)
    assert DeepMNIST().readout_layer([1024, 10], [10], layer_2) is not None
