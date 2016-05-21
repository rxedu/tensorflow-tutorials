# pylint: disable=import-error
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name

import pytest

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_tutorials.mnist import MNIST

@pytest.fixture(scope='session')
def mnist_data(tmpdir_factory):
    data_dir = tmpdir_factory.mktemp('data').join('mnist')
    return input_data.read_data_sets(str(data_dir), one_hot=True)

def test_inputs():
    assert MNIST().inputs is not None

def test_model():
    assert MNIST().model is not None

def test_distribution():
    assert MNIST().distribution is not None

def test_entropy():
    assert MNIST().entropy is not None

def test_train_step():
    assert MNIST().train_step is not None

def test_init():
    assert MNIST().init is not None

def test_train(mnist_data):
    session = MNIST().train(mnist_data.train)
    assert session is not None
    session.close()

def test_check_accuracy(mnist_data):
    mnist = MNIST()
    session = mnist.train(mnist_data.train)
    accuracy = mnist.check_accuracy(mnist_data.test, session)
    assert accuracy >= 0.91
    assert accuracy <= 0.93
    session.close()
