# pylint: disable=import-error
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name

import pytest

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_tutorials.mnist import mnist

@pytest.fixture(scope='session')
def mnist_data(tmpdir_factory):
    data_dir = tmpdir_factory.mktemp('data').join('mnist')
    return input_data.read_data_sets(str(data_dir), one_hot=True)

def test_inputs():
    assert mnist.MNIST().inputs is not None

def test_model():
    assert mnist.MNIST().model is not None

def test_distribution():
    assert mnist.MNIST().distribution is not None

def test_entropy():
    assert mnist.MNIST().entropy is not None

def test_train_step():
    assert mnist.MNIST().train_step is not None

def test_init():
    assert mnist.MNIST().init is not None

def test_train(mnist_data):
    assert mnist.MNIST().train(mnist_data.train) is not None

def test_check_accuracy(mnist_data):
    m = mnist.MNIST() # pylint: disable=invalid-name
    sess = m.train(mnist_data.train)
    accuracy = m.check_accuracy(mnist_data.test, sess)
    assert accuracy >= 0.91
    assert accuracy <= 0.93
