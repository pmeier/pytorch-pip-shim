import contextlib
import functools
import io
import itertools
import re
import subprocess
import sys
from os import path
from sys import platform
from unittest import mock

import pytest

from pip._internal.cli import main
from pip._internal.network.download import _http_get_download
from pip._internal.operations.prepare import RequirementPreparer
from pip._vendor.requests.models import Response
from pip._vendor.urllib3.response import HTTPResponse

import pytorch_pip_shim as pps
from pytorch_pip_shim.computation_backend import ComputationBackend

from tests import mocks, utils


class Package:
    def __init__(self, name, version):
        self.name = name
        self.version = version

    @property
    def specifier(self):
        return f"{self.name}=={self.version}"

    def __eq__(self, other):
        if not isinstance(other, Package):
            return False

        return hash(self) == hash(other)

    def __hash__(self):
        return hash((self.name, self.version))

    def __repr__(self):
        return self.specifier


class PyTorchPackage(Package):
    VALID_NAMES = ("torch", "torchvision", "torchaudio", "torchtext")

    def __init__(self, name, version, computation_backend):
        if name not in self.VALID_NAMES:
            raise RuntimeError

        super().__init__(name, version)

        if isinstance(computation_backend, str):
            computation_backend = ComputationBackend.from_str(computation_backend)
        self.computation_backend = computation_backend

    @classmethod
    def from_regular_package(cls, package, computation_backend):
        return cls(package.name, package.version, computation_backend)

    def __hash__(self):
        return hash((self.name, self.version, self.computation_backend))

    def __repr__(self):
        return f"{self.specifier}+{self.computation_backend}"


def run_pip_cmd(*args):
    with io.StringIO() as stdout_capture, contextlib.redirect_stdout(stdout_capture):
        with mock.patch.object(sys, "argv", new=["pip", *args]):
            main.main()
        return stdout_capture.getvalue().strip()


def pip_install(*requirements, computation_backend=None, cpu=False, upgrade=False):
    flags = []
    if computation_backend is not None:
        flags.append(f"--computation-backend={computation_backend}")
    if cpu:
        flags.append("--cpu")
    if upgrade:
        flags.append("--upgrade")
    return run_pip_cmd(
        "install", *flags, *(requirement.specifier for requirement in requirements)
    )


def pip_uninstall(*packages, yes=True):
    flags = []
    if yes:
        flags.append("--yes")
    return run_pip_cmd(
        "uninstall", *flags, *(package.specifier for package in packages)
    )


DIST_PATTERN = f"(?P<name>({'|'.join(PyTorchPackage.VALID_NAMES)}))"
VERSION_PATTERN = r"(?P<version>[\d.]+)"  # FIXME
COMPUTATION_BACKEND_PATTERN = r"(?P<computation_backend>(cpu|cu\d{2,}))"
FILENAME_PATTERN = re.compile(
    "^"
    + DIST_PATTERN
    + "-"
    + VERSION_PATTERN
    + "([+]"
    + COMPUTATION_BACKEND_PATTERN
    + ")?"
)
URL_PATTERN = re.compile(
    f"^https://download.pytorch.org/whl/{COMPUTATION_BACKEND_PATTERN}"
)

# FIXME: this should be part of the package
TORCH_COMPATIBILITY = {
    Package("torchvision", "0.7.0"): Package("torch", "1.6.0"),
    Package("torchvision", "0.8.1"): Package("torch", "1.7.0"),
}


def wheel_mock(root, name, version):
    head = ["from setuptools import setup", "setup("]
    tail = [")"]

    body = [f"name='{name}'", f"version='{version}'"]
    if name != "torch" and name in PyTorchPackage.VALID_NAMES:
        install_requires = TORCH_COMPATIBILITY[Package(name, version)].specifier
        body.append(f"install_requires=['{install_requires}']")

    lines = "\n".join(head + [f"    {line}," for line in body] + tail)

    with open(path.join(root, "setup.py"), "w") as fh:
        fh.writelines(lines)

    subprocess.check_call(
        f"{sys.executable} setup.py bdist_wheel", shell=True, cwd=root
    )

    return path.join(root, "dist", f"{name}-{version}-py3-none-any.whl")


@pytest.fixture(autouse=True)
def disable_pytorch_download(mocker, tmpdir):
    prepare_linked_requirement = RequirementPreparer.prepare_linked_requirement

    pytorch_wheels = {}

    def new_prepare_linked_requirement(self, req, parallel_builds=False):
        old = functools.partial(prepare_linked_requirement, self, req, parallel_builds)

        match = FILENAME_PATTERN.match(req.link.filename)
        if match is None:
            return old()

        name = match.group("name")
        version = match.group("version")
        with open(wheel_mock(tmpdir, name, version), "rb") as fh:
            pytorch_wheels[req.link] = fh.read()

        return old()

    mocker.patch(
        "pip._internal.operations.prepare.RequirementPreparer.prepare_linked_requirement",
        new=new_prepare_linked_requirement,
    )

    def mocked_response(content):
        response = Response()
        response.headers["content-length"] = len(content)

        body = io.BytesIO()
        body.write(content)
        raw = HTTPResponse(body=body)
        response.raw = raw
        body.seek(0)

        return response

    def new_http_get_download(session, link):
        try:
            return mocked_response(pytorch_wheels[link])
        except KeyError:
            return _http_get_download(session, link)

    mocker.patch(
        "pip._internal.network.download._http_get_download", new=new_http_get_download
    )


@contextlib.contextmanager
def disable_installation():
    installed_pytorch_packages = set()

    def new(install_req, *args, **kwargs):
        match = FILENAME_PATTERN.match(install_req.link.filename)
        if match is None:
            return

        attrs = {
            attr: match.group(attr)
            for attr in ("name", "version", "computation_backend")
        }

        if attrs["computation_backend"] is None:
            match = URL_PATTERN.match(install_req.link.url)
            if match is not None:
                attrs["computation_backend"] = match.group("computation_backend")

        if attrs["computation_backend"] is not None:
            cls = PyTorchPackage
        else:
            attrs.pop("computation_backend")
            cls = Package

        installed_pytorch_packages.add(cls(**attrs))

    with mock.patch(
        "pip._internal.req.req_install.InstallRequirement.install", new=new
    ):
        yield installed_pytorch_packages


class InstallParameter:
    # FIXME: populate this automatically
    AVAILABLE_COMPUTATION_BACKENDS = {
        "1.6.0": ("cpu", "cu92", "cu101", "cu102"),
        "1.7.0": ("cpu", "cu92", "cu101", "cu102", "cu110"),
    }

    def __init__(
        self, requirement: Package, expected_dists=None, computation_backends=None
    ):
        self.requirement = requirement
        self.expected_dists = self.parse_expected_dists(expected_dists)
        self.computation_backends = self.parse_computation_backends(
            computation_backends
        )

    def parse_expected_dists(self, expected_dists):
        if expected_dists is None:
            return set()
        elif isinstance(expected_dists, Package):
            return {expected_dists}
        else:
            return set(expected_dists)

    def parse_computation_backends(self, computation_backends):
        if computation_backends is None:
            if platform == "darwin":
                computation_backends = "cpu"
            else:
                version = next(
                    (
                        dist.version
                        for dist in self.expected_dists
                        if dist.name == "torch"
                    )
                )
                computation_backends = self.AVAILABLE_COMPUTATION_BACKENDS.get(
                    version, "cpu"
                )

        if isinstance(computation_backends, (str, ComputationBackend)):
            computation_backends = [computation_backends]

        if platform.startswith("win"):
            computation_backends = list(computation_backends)
            computation_backends.remove("cu92")

        return [
            ComputationBackend.from_str(computation_backend)
            if isinstance(computation_backend, str)
            else computation_backend
            for computation_backend in computation_backends
        ]

    def argvalues_and_ids(self):
        for computation_backend in self.computation_backends:
            expected_pytorch_dists = {
                PyTorchPackage.from_regular_package(dist, computation_backend)
                for dist in self.expected_dists
            }

            argvalue = (self.requirement, computation_backend, expected_pytorch_dists)
            id = f"{self.requirement}, {computation_backend}"
            yield argvalue, id


def install_parametrization(*install_parameters):
    argnames = ("requirement", "computation_backend", "expected_pytorch_dists")
    argvalues, ids = zip(
        *itertools.chain(
            *(
                install_parameter.argvalues_and_ids()
                for install_parameter in install_parameters
            )
        )
    )
    return pytest.mark.parametrize(
        argnames,
        argvalues=argvalues,
        ids=ids,
    )


@pytest.mark.skipif(
    platform == "darwin",
    reason=(
        "Test infrastructure cannot handle packages "
        "without computation backends yet."
    ),
)
@utils.skip_if_pip_is_not_shimmed
@pytest.mark.slow
@install_parametrization(
    InstallParameter(
        Package("torch", "1.6.0"),
        Package("torch", "1.6.0"),
    ),
    InstallParameter(
        Package("torch", "1.7.0"),
        Package("torch", "1.7.0"),
    ),
    InstallParameter(
        Package("torchvision", "0.8.1"),
        (Package("torch", "1.7.0"), Package("torchvision", "0.8.1")),
    ),
    InstallParameter(
        Package("pystiche", "0.6.3"),
        (Package("torch", "1.6.0"), Package("torchvision", "0.7.0")),
    ),
)
def test_install(requirement: Package, computation_backend, expected_pytorch_dists):
    with disable_installation() as installed_pytorch_dists:
        pip_install(
            requirement,
            computation_backend=computation_backend,
        )

    assert installed_pytorch_dists == expected_pytorch_dists


def test_uninstall(mocker):
    mock = mocker.patch(mocks.make_target("shim", "remove"))

    # although not specifically mocked, this will not uninstall pytorch-pip-shim since
    # it is outside the environment we are currently running in
    pip_uninstall(Package("pytorch-pip-shim", pps.__version__))

    mock.assert_called()
