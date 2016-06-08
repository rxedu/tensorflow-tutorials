"""
[Tutorials] for [TensorFlow].

[Source on GitHub][source].

[source]: https://github.com/rxedu/tensorflow-tutorials
[Tutorials]: https://www.tensorflow.org/versions/r0.9/tutorials/index.html
[TensorFLow]: https://www.tensorflow.org/
"""
from .mnist import mnist
from .deep_mnist import deep_mnist

__all__ = [
    'mnist',
    'deep_mnist'
]
