from setuptools import find_packages, setup

from tensorflow_tutorials import __version__

with open('README.rst', 'r') as f:
    long_description = f.read()

setup(
    name='tensorflow-tutorials',
    version=__version__,
    author='Evan Sosenko',
    author_email='razorx@evansosenko.com',
    packages=find_packages(exclude=['docs']),
    url='https://github.com/rxedu/tensorflow-tutorials',
    license='MIT',
    description='Tutorials for TensorFlow.',
    long_description=long_description,
    install_requires=[
    ]
)
