import contextlib
import functools
import itertools
import subprocess
import sys

import pytest

from pip._internal.cli import main as pip_cli

import pytorch_pip_shim
from pytorch_pip_shim import cli as pps_cli
from pytorch_pip_shim.utils import canocialize_name

from tests import mocks


@contextlib.contextmanager
def exits_correctly(code=None, error=False, must_exit=True):
    try:
        yield
    except SystemExit as exit:
        returned_code = exit.code
    else:
        if must_exit:
            raise AssertionError("Program did not exit.")
        else:
            return

    if code is not None:
        assert returned_code == code
    else:
        assert bool(returned_code) is error


def run_main(mocker, capsys, cli, *args, name=None, **exit_kwargs):
    if name is None:
        name = cli.__name__
    mocker.patch.object(sys, "argv", [name, *args])

    with exits_correctly(**exit_kwargs):
        cli.main()

    out, _ = capsys.readouterr()
    return out.strip()


@pytest.fixture
def pps_main(mocker, capsys):
    return functools.partial(run_main, mocker, capsys, pps_cli, name="pytorch-pip-shim")


def test_pps_main_smoke():
    subprocess.check_call("python -m pytorch_pip_shim -h", shell=True)


def test_pps_unknown_subcommand(pps_main):
    pps_main("unknown", error=True)


def test_pps_unknown_option(pps_main):
    pps_main("--unknown", error=True)


@pytest.mark.parametrize("arg", ["-h", "--help"])
def test_pps_help_smoke(pps_main, arg):
    pps_main(arg)


@pytest.mark.parametrize("arg", ["-V", "--version"])
def test_version(pps_main, arg):
    out = pps_main(arg)

    name = canocialize_name(pytorch_pip_shim.__name__)
    version = pytorch_pip_shim.__version__
    path = pytorch_pip_shim.__path__[0]

    assert f"{name}=={version} from {path}" == out


def test_insert(mocker, pps_main):
    mock = mocker.patch(mocks.make_target("shim", "insert"))

    pps_main("insert")

    mock.assert_called_once()


def test_insert_with_file(mocker, pps_main):
    file = "file"
    mock = mocker.patch(mocks.make_target("shim", "insert"))

    pps_main("insert", file)

    mock.assert_called_once_with(file)


def test_remove(mocker, pps_main):
    mock = mocker.patch(mocks.make_target("shim", "remove"))

    pps_main("remove")

    mock.assert_called_once()


def test_remove_with_file(mocker, pps_main):
    file = "file"
    mock = mocker.patch(mocks.make_target("shim", "remove"))

    pps_main("remove", file)

    mock.assert_called_once_with(file)


def parametrize_test_status():
    argname = "is_inserted"
    argvalues = [True, False]
    ids = [f"shim {'is' if argvalue else 'is not'} inserted" for argvalue in argvalues]
    return pytest.mark.parametrize(argname, argvalues, ids=ids)


@parametrize_test_status()
def test_status(mocker, pps_main, is_inserted):
    mocker.patch(mocks.make_target("shim", "is_inserted"), return_value=is_inserted)

    pps_main("status", error=not is_inserted)


def test_status_stdout_smoke(mocker, pps_main):
    mocker.patch(mocks.make_target("shim", "is_inserted"), return_value=True)

    out = pps_main("status")

    assert out


def test_status_with_file(mocker, pps_main):
    file = "file"
    mock = mocker.patch(mocks.make_target("shim", "is_inserted"))

    pps_main("status", file)

    mock.assert_called_once_with(file)


@pytest.mark.parametrize("arg", ["-q", "--quiet"])
def test_status_quiet(mocker, pps_main, arg):
    mocker.patch(mocks.make_target("shim", "is_inserted"), return_value=True)

    out = pps_main("status", arg)

    assert not out


def test_detect(mocker, pps_main, generic_computation_backend):
    mocker.patch(
        mocks.make_target("cli", "commands", "detect"),
        return_value=generic_computation_backend,
    )

    out = pps_main("detect")

    assert out == str(generic_computation_backend)


@pytest.fixture
def pip_main(mocker, capsys):
    return functools.partial(run_main, mocker, capsys, pip_cli, name="pip")


def parametrize_pip_help_install():
    opts = ("--computation-backend", "--cpu")
    argss = (("help", "install"), ("install", "--help"))

    argnames = ("args", "option")
    argvalues = tuple(itertools.product(argss, opts))
    ids = [f"{' '.join(('pip', *args))}, {opt}" for args, opt in argvalues]
    return pytest.mark.parametrize(argnames, argvalues, ids=ids)


@parametrize_pip_help_install()
def test_pip_help_install_computation_backend(pip_main, args, option):
    out = pip_main(*args, must_exit=False)

    assert option in out
