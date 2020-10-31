import optparse

import pytest

from pytorch_pip_shim import computation_backend as cb
from pytorch_pip_shim import utils

from tests import mocks


def test_canonicalize_name():
    assert utils.canocialize_name("Pytorch_PIP-shim") == "pytorch-pip-shim"


def make_computation_backend_opts(computation_backend=None, cpu=False):
    opts = optparse.Values()
    opts.computation_backend = computation_backend
    opts.cpu = cpu
    return opts


def test_process_computation_backend(mocker, generic_computation_backend):
    string = "generic"
    mocker.patch(
        mocks.make_target("computation_backend", "ComputationBackend", "from_str"),
        return_value=generic_computation_backend,
    )
    opts = make_computation_backend_opts(computation_backend=string)

    assert utils.process_computation_backend(opts) == generic_computation_backend


def test_process_computation_backend_cpu(generic_computation_backend):
    opts = make_computation_backend_opts(cpu=True)

    assert utils.process_computation_backend(opts) == cb.CPUBackend()


def test_process_computation_backend_detect(
    mocker,
    generic_computation_backend,
):
    mocker.patch(
        mocks.make_target("computation_backend", "detect"),
        return_value=generic_computation_backend,
    )
    opts = make_computation_backend_opts()

    assert utils.process_computation_backend(opts) == generic_computation_backend


def test_import_fn():
    with pytest.raises(utils.InternalError):
        utils.import_fn("")
