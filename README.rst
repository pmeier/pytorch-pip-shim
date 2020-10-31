``pytorch-pip-shim``
====================

.. start-badges

.. list-table::
    :stub-columns: 1

    * - package
      - |license| |status|
    * - code
      - |isort| |black| |mypy| |lint|
    * - tests
      - |tests| |coverage|

.. end-badges

- `What is it?`_
- `Why do I need it?`_
- `How do I install it?`_
- `How do I use it?`_
- `How do I uninstall it?`_
- `How do I configure it?`_
- `How does it work?`_

Disclaimer
==========

Neither this project (``pytorch-pip-shim``) nor its author
(`Philip Meier <https://github.com/pmeier>`_) are affiliated with
`PyTorch <https://pytorch.org>`_ in any way. PyTorch and any related
marks are
`trademarks of Facebook, Inc <https://pytorch.org/assets/brand-guidelines/PyTorch-Brand-Guidelines.pdf>`_.

What is it?
===========

``pytorch-pip-shim`` is a small background utility that eases the installation process
with ``pip`` for PyTorch and third-party packages that depend on its distributions.
After the shim is inserted, you can install PyTorch with ``pip`` like you do with any
other package.

Why do I need it?
=================

PyTorch is fully ``pip install`` able, but PyPI, the default ``pip`` search index, has
some limitations:

1. PyPI regularly only allows binaries up to a size of
   `approximately 60 MB <https://github.com/pypa/packaging-problems/issues/86>`_. You
   can `request a file size limit increase <https://pypi.org/help/#file-size-limit>`_
   (and the PyTorch team probably did that), but it is still not enough: the Windows
   binaries cannot be installed through `PyPI <https://pypi.org/project/torch/#files>`_
   due to their size.
2. PyTorch uses local version specifiers to indicate for which computation backend the
   binary was compiled, for example ``torch==1.6.0+cpu``. Unfortunately, local
   specifiers are not allowed on PyPI. Thus, only the binaries compiled with the latest
   CUDA version are uploaded. If you do not have a CUDA capable GPU, downloading this
   is only a waste of bandwidth and disk capacity. If on the other hand simply don't
   have the latest CUDA driver installed, you can't use any of the GPU features.

To overcome this, PyTorch alos hosts *all* binaries
`themselves <https://download.pytorch.org/whl/torch_stable.html>`_. To access them, you
still can use ``pip install``, but have to use some
`additional options <https://pytorch.org/get-started/locally/>`_:

.. code-block:: sh

  $ pip install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

While this is certainly an improvement, it also has it downside: in addition to the
computation backend, the version has to be specified exactly. Without knowing what the
latest release is, it is impossible to install it as simple as ``pip install torch``
normally would.

At this point you might justifiably as: why don't you just use ``conda`` as PyTorch
recommends?

.. code-block:: sh

  $ conda install pytorch cpuonly -c pytorch

This should cover all cases, right? Well, almost. The above command is enough if you
just need PyTorch. Imagine the case of a package that depends on PyTorch, but
cannot be installed with ``conda`` since it is hosted on PyPI? You can't use the ``-f``
option since the package in question is not hosted by PyTorch. Thus, you now have to
manually track down (and resolve in the case of multiple packages) the PyTorch
distributions, install them in a first step and only install the actual package (and
all other dependencies) afterwards.

If just want to use ``pip install`` like you always did before without worrying about
any of the stuff above, ``pytorch-pip-shim`` was made for you.

How do I install it?
====================

Installing ``pytorch-pip-shim`` is as easy as

.. code-block:: sh

  $ pip install pytorch-pip-shim

Since it depends on ``pip`` and it might be upgraded during installation,
`Windows users <https://pip.pypa.io/en/stable/installing/#upgrading-pip>`_ should
install it with

.. code-block:: sh

  $ python -m pip install pytorch-pip-shim

How do I use it?
================

After ``pytorch-pip-shim`` is installed there is only a single step to insert the shim:

.. code-block:: sh

  $ pytorch-pip-shim insert

After that you can use ``pip`` as you did before and ``pytorch-pip-shim`` handles the
computation backend auto-detection for you in the background.

If you want to remove the shim you can do so with

.. code-block:: sh

  $ pytorch-pip-shim remove

You can check its status with

.. code-block:: sh

  $ pytorch-pip-shim status

How do I uninstall it?
======================

Uninstalling is as easy as

.. code-block:: sh

  $ pip uninstall pytorch-pip-shim

By doing so, ``pytorch-pip-shim`` automatically removes the shim if inserted.

How do I configure it?
======================

Once inserted, you don't need to configure anything. If you don't want the computation
backend auto-detected but rather want to set it manually ``pytorch-pip-shim`` adds two
CLI options to ``pip install``:

- ``--computation-backend <computation_backend>``
- ``--cpu``

How does it work?
=================

The authors of ``pip`` **do not condone** the use of ``pip`` internals as they might
break without warning. As a results of this, ``pip`` has no capability for plugins to
hook into specific tasks. Thus, the only way to patch ``pip`` s functionality is to
adapt its source in-place. Although this is really bad practice, it is unavoidable for
the goal of this package.

``pystiche-pip-shim`` inserts a shim into the ``pip`` main file, which decorates the
main function. Everytime you call ``pip install``, some aspects of the installation
process are patched:

- While searching for a download link for a PyTorch distribution, ``pytorch-pip-shim``
  replaces the default search index. This is equivalent to calling ``pip install`` with
  the ``-f`` option only for PyTorch distributions.
- While evaluating possible PyTorch installation candidates, ``pytorch-pip-shim`` culls
  binaries not compatible with the available hardware.

.. |license|
  image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: License

.. |status|
  image:: https://www.repostatus.org/badges/latest/wip.svg
    :alt: Project Status: WIP
    :target: https://www.repostatus.org/#wip

.. |isort|
  image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
    :target: https://timothycrosley.github.io/isort/
    :alt: isort

.. |black|
  image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black
   
.. |mypy|
  image:: http://www.mypy-lang.org/static/mypy_badge.svg
    :target: http://mypy-lang.org/
    :alt: mypy

.. |lint|
  image:: https://github.com/pmeier/pytorch-pip-shim/workflows/lint/badge.svg
    :target: https://github.com/pmeier/pytorch-pip-shim/actions?query=workflow%3Alint+branch%3Amaster
    :alt: Lint status via GitHub Actions

.. |tests|
  image:: https://github.com/pmeier/pytorch-pip-shim/workflows/tests/badge.svg
    :target: https://github.com/pmeier/pytorch-pip-shim/actions?query=workflow%3Atests+branch%3Amaster
    :alt: Test status via GitHub Actions

.. |coverage|
  image:: https://codecov.io/gh/pmeier/pytorch-pip-shim/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pmeier/pytorch-pip-shim
    :alt: Test coverage via codecov.io
