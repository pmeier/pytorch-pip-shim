import pytest

from pytorch_pip_shim import computation_backend as cb


class GenericComputationBackend(cb.ComputationBackend):
    @property
    def local_specifier(self):
        return "generic"


@pytest.fixture
def generic_computation_backend():
    return GenericComputationBackend()
