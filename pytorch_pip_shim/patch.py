import contextlib
import functools
import optparse
import re
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Text, Tuple, cast
from unittest import mock
from urllib.parse import urljoin

from pip._internal.index.collector import LinkCollector
from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.req.req_uninstall import UninstallPathSet

from . import shim
from .computation_backend import ComputationBackend
from .utils import apply_patch, computation_backend_options, parse_pip_args

__all__ = ["patch"]

PATCHED_SUB_CMDS = ("install", "uninstall")
PYTORCH_DISTRIBUTIONS = ("torch", "torchvision", "torchaudio", "torchtext")


def patch(pip_main: Callable[..., int]) -> Callable[..., int]:
    @functools.wraps(pip_main)
    def shim(args: Optional[List[str]] = None) -> int:
        if args is None:
            args = sys.argv[1:]

        with apply_patches(args):
            return pip_main(args=args)

    return shim


@contextlib.contextmanager
def apply_patches(args: List[str]) -> Iterator[contextlib.ExitStack]:
    args = parse_pip_args(args)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch_cli_options())
        stack.enter_context(
            patch_link_collection(args.computation_backend, args.nightly)
        )
        stack.enter_context(patch_link_evaluation())
        stack.enter_context(patch_candidate_selection(args.computation_backend))
        stack.enter_context(patch_self_uninstallation())
        yield stack


@contextlib.contextmanager
def patch_cli_options() -> Iterator[None]:
    def postprocessing(
        args: Tuple[optparse.OptionGroup], kwargs: Dict[str, Any], output: None
    ) -> None:
        (cmd_opts,) = args

        for option in computation_backend_options():
            cmd_opts.add_option(option)

    with apply_patch(
        "pip._internal.cli.cmdoptions.add_target_python_options",
        postprocessing=postprocessing,  # type: ignore[arg-type]
    ):
        yield


@contextlib.contextmanager
def patch_link_collection(
    computation_backend: ComputationBackend, nightly: bool
) -> Iterator[None]:
    base = "https://download.pytorch.org/whl/"
    url = (
        f"nightly/{computation_backend}/torch_nightly.html"
        if nightly
        else "torch_stable.html"
    )
    search_scope = SearchScope.create([urljoin(base, url)], [])

    @contextlib.contextmanager
    def context(args: Tuple[LinkCollector, str], kwargs: Any) -> Iterator[None]:
        self, project_name, *_ = args
        if project_name not in PYTORCH_DISTRIBUTIONS:
            yield
            return

        with mock.patch.object(self, "search_scope", search_scope):
            yield

    with apply_patch(
        "pip._internal.index.collector.LinkCollector.collect_links",
        context=context,  # type: ignore[arg-type]
    ):
        yield


@contextlib.contextmanager
def patch_link_evaluation() -> Iterator[None]:
    HAS_LOCAL_PATTERN = re.compile(r"[+](cpu|cu\d+)$")
    COMPUTATION_BACKEND_PATTERN = re.compile(
        r"^/whl/(?P<computation_backend>(cpu|cu\d+))/"
    )

    def postprocessing(
        args: Tuple[Any, Link],
        kwargs: Any,
        output: Tuple[bool, Optional[Text]],
    ) -> Tuple[bool, Optional[Text]]:
        _, link = args
        is_candidate, result = output
        if not is_candidate:
            return output

        result = cast(Text, result)
        has_local = HAS_LOCAL_PATTERN.search(result) is not None
        if has_local:
            return output

        computation_backend = COMPUTATION_BACKEND_PATTERN.match(link.path)
        if not computation_backend:
            return output

        return True, f"{result}+{computation_backend.group('computation_backend')}"

    with apply_patch(
        "pip._internal.index.package_finder.LinkEvaluator.evaluate_link",
        postprocessing=postprocessing,  # type: ignore[arg-type]
    ):
        yield


@contextlib.contextmanager
def patch_candidate_selection(
    computation_backend: ComputationBackend,
) -> Iterator[None]:
    def postprocessing(
        args: Any,
        kwargs: Any,
        output: List[InstallationCandidate],
    ) -> List[InstallationCandidate]:
        return [
            candidate
            for candidate in output
            if candidate.name not in PYTORCH_DISTRIBUTIONS
            or candidate.version.local is None
            or candidate.version.local == computation_backend
        ]

    with apply_patch(
        "pip._internal.index.package_finder.CandidateEvaluator.get_applicable_candidates",
        postprocessing=postprocessing,
    ):
        yield


@contextlib.contextmanager
def patch_self_uninstallation() -> Iterator[None]:
    def preprocessing(
        args: Tuple[UninstallPathSet, ...], kwargs: Any
    ) -> Tuple[Tuple[UninstallPathSet, ...], Any]:
        self, *_ = args
        if self.dist.project_name == "pytorch-pip-shim":
            shim.remove()
        return args, kwargs

    with apply_patch(
        "pip._internal.req.req_uninstall.UninstallPathSet.remove",
        preprocessing=preprocessing,
    ):
        yield
