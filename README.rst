TensorFlow Tutorials
=======================

|GitHub-license| |Requires.io| |Travis| |Codecov|

    Built from `makenew/python-package <https://github.com/makenew/python-package>`__.

.. |GitHub-license| image:: https://img.shields.io/github/license/rxedu/tensorflow-tutorials.svg
   :target: ./LICENSE.txt
   :alt: GitHub license
.. |Requires.io| image:: https://img.shields.io/requires/github/rxedu/tensorflow-tutorials.svg
   :target: https://requires.io/github/rxedu/tensorflow-tutorials/requirements/
   :alt: Requires.io
.. |Travis| image:: https://img.shields.io/travis/rxedu/tensorflow-tutorials.svg
   :target: https://travis-ci.org/rxedu/tensorflow-tutorials
   :alt: Travis
.. |Codecov| image:: https://img.shields.io/codecov/c/github/rxedu/tensorflow-tutorials.svg
   :target: https://codecov.io/gh/rxedu/tensorflow-tutorials
   :alt: Codecov

Description
-----------

`Tutorials for TensorFlow`_.

.. _Tutorials for TensorFlow: https://www.tensorflow.org/versions/r0.8/tutorials/index.html

Requirements
------------

- Python 3.6 (tested on Linux 64-bit).
- A TensorFlow_ distribution appropriate to your environment.

.. _TensorFLow: https://www.tensorflow.org/

Installation
------------

Add this line to your application's requirements.txt

::

    https://github.com/rxedu/tensorflow-tutorials/archive/master.zip

and install it with

::

    $ pip install -r requirements.txt

If you are writing a Python package which will depend on this,
add this to your requirements in ``setup.py``.

Alternatively, install it directly using pip with

::

    $ pip install https://github.com/rxedu/tensorflow-tutorials/archive/master.zip

Development and Testing
-----------------------

Source Code
~~~~~~~~~~~

The `tensorflow-tutorials source`_ is hosted on GitHub.
Clone the project with

::

    $ git clone https://github.com/rxedu/tensorflow-tutorials.git

.. _tensorflow-tutorials source: https://github.com/rxedu/tensorflow-tutorials

Requirements
~~~~~~~~~~~~

You will need `Python 3`_ with pip_.

Install the development dependencies with

::

    $ pip install -r requirements.devel.txt

.. _pip: https://pip.pypa.io/
.. _Python 3: https://www.python.org/

Tests
~~~~~

Lint code with

::

    $ python setup.py lint


Run tests with

::

    $ python setup.py test

Run tests automatically on changes with

::

    $ ptw

Documentation
~~~~~~~~~~~~~

Generate documentation with

::

    $ make docs


Publish to GitHub Pages with

::

    $ make gh-pages

Contributing
------------

Please submit and comment on bug reports and feature requests.

To submit a patch:

1. Fork it (https://github.com/rxedu/tensorflow-tutorials/fork).
2. Create your feature branch (``git checkout -b my-new-feature``).
3. Make changes. Write and run tests.
4. Commit your changes (``git commit -am 'Add some feature'``).
5. Push to the branch (``git push origin my-new-feature``).
6. Create a new Pull Request.

License
-------

This Python package is licensed under the MIT license.

Warranty
--------

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall the copyright holder or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused and on
any theory of liability, whether in contract, strict liability, or tort
(including negligence or otherwise) arising in any way out of the use of this
software, even if advised of the possibility of such damage.
