Contributing guide lines
========================

We appreciate all contributions. If you are planning to contribute bug-fixes or
documentation improvements, please open a
`pull request (PR) <https://github.com/pmeier/pytorch-pip-shim/pulls>`_
without further discussion. If you planning to contribute new features, please open an
`issue <https://github.com/pmeier/pytorch-pip-shim/issues>`_
and discuss the feature with us first.

To start working on ``pytorch-pip-shim`` clone from the latest version and install 
the development requirements:

.. code-block:: sh

  PYTORCH-PIP-SHIM_ROOT = pytorch-pip-shim
  git clone https://github.com/pmeier/pytorch-pip-shim $PYTORCH-PIP-SHIM_ROOT
  cd $PYTORCH-PIP-SHIM_ROOT
  pip install -r requirements-dev.txt
  pre-commit install

Every PR is subjected to multiple checks that it has to pass before it can be merged.
The checks are performed by `tox <https://tox.readthedocs.io/en/latest/>`_ . Below
you can find details and instructions how to run the checks locally.


Code format and linting
-----------------------

``pytorch-pip-shim`` uses `isort <https://timothycrosley.github.io/isort/>`_ to sort the
imports, `black <https://black.readthedocs.io/en/stable/>`_ to format the code, and
`flake8 <https://flake8.pycqa.org/en/latest/>`_ to enforce
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ compliance.

Furthermore, ``pytorch-pip-shim`` is `PEP561 <https://www.python.org/dev/peps/pep-0561/>`_
compliant and checks the type annotations with `mypy <http://mypy-lang.org/>`_ .

To format your code run

.. code-block:: sh

  cd $PYTORCH-PIP-SHIM_ROOT
  tox -e format

.. note::

  Amongst others, ``isort`` and ``black`` are run by
  `pre-commit <https://pre-commit.com/>`_ before every commit.

To run the full lint check locally run

.. code-block:: sh

  cd $PYTORCH-PIP-SHIM_ROOT
  tox -e lint


Tests
-----

``pytorch-pip-shim`` uses `pytest <https://docs.pytest.org/en/stable/>`_ to run
the test suite. You can run it locally with

.. code-block:: sh

  cd $PYTORCH-PIP-SHIM_ROOT
  tox

.. note::

  ``pytorch-pip-shim`` adds the following custom options with the
  corresponding ``@pytest.mark.*`` decorators:
  - ``--skip-large-download``: ``@pytest.mark.large_download``
  - ``--skip-slow``: ``@pytest.mark.slow``
  - ``--run-flaky``: ``@pytest.mark.flaky``

  Options prefixed with ``--skip`` are run by default and skipped if the option is
  given. Options prefixed with ``--run`` are skipped by default and run if the option
  is given.

  These options are passed through ``tox`` if given after a ``--`` flag. For example,
  the CI invocation command is equivalent to:

  .. code-block:: sh

    cd $PYTORCH-PIP-SHIM_ROOT
    tox -- --skip-large-download


Documentation
-------------

To build the html and latex documentation locally, run

.. code-block:: sh

  cd $PYTORCH-PIP-SHIM_ROOT
  tox -e docs
