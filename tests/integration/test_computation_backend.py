import subprocess

import pytest

from pytorch_pip_shim import computation_backend as cb

from tests import mocks, utils


def test_ComputationBackend_eq(generic_computation_backend):
    assert generic_computation_backend == generic_computation_backend
    assert generic_computation_backend == generic_computation_backend.local_specifier
    assert generic_computation_backend != 0


def test_ComputationBackend_hash_smoke(generic_computation_backend):
    assert isinstance(hash(generic_computation_backend), int)


def test_ComputationBackend_repr_smoke(generic_computation_backend):
    assert isinstance(repr(generic_computation_backend), str)


def test_ComputationBackend_from_str_cpu():
    string = "cpu"
    backend = cb.ComputationBackend.from_str(string)
    assert isinstance(backend, cb.CPUBackend)


def parametrize_test_ComputationBackend_from_str_cuda():
    major, minor = 12, 3
    strings = (
        f"cu{major}{minor}",
        f"cu{major}.{minor}",
        f"cuda{major}{minor}",
        f"cuda{major}.{minor}",
    )
    argnames = ("string", "major", "minor")
    argvalues = tuple(zip(strings, [major] * len(strings), [minor] * len(strings)))
    ids = strings
    return pytest.mark.parametrize(argnames, argvalues, ids=ids)


@parametrize_test_ComputationBackend_from_str_cuda()
def test_ComputationBackend_from_str_cuda(string, major, minor):
    backend = cb.ComputationBackend.from_str(string)
    assert isinstance(backend, cb.CUDABackend)
    assert backend.major == major
    assert backend.minor == minor


@pytest.mark.parametrize("string", ("unknown", "cudnn"))
def test_ComputationBackend_from_str_unknown(string):
    with pytest.raises(cb.ParseError):
        cb.ComputationBackend.from_str(string)


def test_CPUBackend():
    backend = cb.CPUBackend()
    assert backend == "cpu"


def test_CUDABackend():
    major = 42
    minor = 21
    backend = cb.CUDABackend(major, minor)
    assert backend == f"cu{major}{minor}"


@pytest.fixture
def patch_nvcc_call(mocker):
    def patch_nvcc_call_(**kwargs):
        return mocker.patch(
            mocks.make_target("computation_backend.subprocess.check_output"),
            **kwargs,
        )

    return patch_nvcc_call_


def test_detect_computation_backend_no_nvcc(patch_nvcc_call):
    patch_nvcc_call(side_effect=subprocess.CalledProcessError(1, ""))

    assert isinstance(cb.detect(), cb.CPUBackend)


def test_detect_computation_backend_unknown_release(patch_nvcc_call):
    patch_nvcc_call(return_value="release unknown".encode("utf-8"))

    assert isinstance(cb.detect(), cb.CPUBackend)


def test_detect_computation_backend_cuda(patch_nvcc_call):
    major = 42
    minor = 21
    patch_nvcc_call(
        return_value=f"foo\nbar, release {major}.{minor}, baz".encode("utf-8")
    )

    backend = cb.detect()
    assert isinstance(backend, cb.CUDABackend)
    assert backend.major == major
    assert backend.minor == minor


@utils.skip_if_cuda_unavailable
def test_detect_computation_backend_cuda_smoke():
    assert isinstance(cb.detect(), cb.CUDABackend)
