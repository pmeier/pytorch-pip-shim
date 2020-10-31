import hashlib
import importlib.util
from os import path
from urllib.request import urlopen

import pytest

import pip

from pytorch_pip_shim import shim

from tests.mocks import patch_imports


def compute_md5_checksum(file: str):
    with open(file, "rb") as bfh:
        return hashlib.md5(bfh.read()).hexdigest()


def make_pip_main_file(root, content, name="pps_main.py"):
    file = path.join(root, name)
    with open(file, "w") as fh:
        fh.write(content)

    return file, compute_md5_checksum(file)


# TODO: move to utils
def load_pip_main_module_from_file(file):
    spec = importlib.util.spec_from_file_location("pip_main", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def patch_default_pip_main_file(mocker):
    def patch_default_pip_main_file_(file):
        return mocker.patch.object(shim, "FILE", file)

    return patch_default_pip_main_file_


@pytest.fixture
def mocked_unshimmed_pip_main_file(tmpdir):
    return make_pip_main_file(tmpdir, "def pps_main(): pass")


@pytest.fixture
def mocked_shimmed_pip_main_file(tmpdir):
    return make_pip_main_file(
        tmpdir, "# -*- pytorch-pip-shim -*-\ndef pps_main(): pass"
    )


@pytest.fixture(scope="module")
def make_unshimmed_pip_main_file():
    url = f"https://raw.githubusercontent.com/pypa/pip/{pip.__version__}/src/pip/_internal/cli/main.py"
    with urlopen(url) as response:
        content = response.read().decode("utf-8").strip()

    def make(root, name="pps_main.py"):
        return make_pip_main_file(root, content, name=name)

    return make


@pytest.fixture
def unshimmed_pip_main_file(make_unshimmed_pip_main_file, tmpdir):
    return make_unshimmed_pip_main_file(tmpdir)


@pytest.fixture
def shimmed_pip_main_file(unshimmed_pip_main_file):
    file, _ = unshimmed_pip_main_file
    shim.insert(file)
    return file, compute_md5_checksum(file)


def test_is_inserted(patch_default_pip_main_file, mocked_shimmed_pip_main_file):
    file, _ = mocked_shimmed_pip_main_file
    patch_default_pip_main_file(file)

    assert shim.is_inserted()


def test_is_inserted_with_file(mocked_shimmed_pip_main_file):
    file, _ = mocked_shimmed_pip_main_file

    assert shim.is_inserted(file)


def test_is_not_inserted(patch_default_pip_main_file, mocked_unshimmed_pip_main_file):
    file, _ = mocked_unshimmed_pip_main_file
    patch_default_pip_main_file(file)

    assert not shim.is_inserted()


def test_is_not_inserted_with_file(mocked_unshimmed_pip_main_file):
    file, _ = mocked_unshimmed_pip_main_file
    assert not shim.is_inserted(file)


def test_insert(patch_default_pip_main_file, unshimmed_pip_main_file):
    file, _ = unshimmed_pip_main_file
    patch_default_pip_main_file(file)

    shim.insert()
    assert shim.is_inserted()


def test_insert_with_file(unshimmed_pip_main_file):
    file, _ = unshimmed_pip_main_file

    shim.insert(file)
    assert shim.is_inserted(file)


def test_insert_into_already_shimmed_file(shimmed_pip_main_file):
    file, md5 = shimmed_pip_main_file

    shim.insert(file)
    assert compute_md5_checksum(file) == md5


def test_remove(
    patch_default_pip_main_file, shimmed_pip_main_file, unshimmed_pip_main_file
):
    file, _ = shimmed_pip_main_file
    _, md5 = unshimmed_pip_main_file
    patch_default_pip_main_file(file)

    shim.remove()
    assert not shim.is_inserted()
    assert compute_md5_checksum(file) == md5


def test_remove_with_file(shimmed_pip_main_file, unshimmed_pip_main_file):
    file, _ = shimmed_pip_main_file
    _, md5 = unshimmed_pip_main_file

    shim.remove(file)
    assert not shim.is_inserted(file)
    assert compute_md5_checksum(file) == md5


def test_remove_from_unshimmed_file(unshimmed_pip_main_file):
    file, md5 = unshimmed_pip_main_file

    shim.remove(file)
    assert compute_md5_checksum(file) == md5


def test_fallback(mocker, shimmed_pip_main_file):
    with patch_imports("pytorch_pip_shim"):
        file, _ = shimmed_pip_main_file
        module = load_pip_main_module_from_file(file)

    mock = mocker.patch("pytorch_pip_shim.patch")

    with pytest.raises(SystemExit):
        module.main()

    mock.assert_not_called()
