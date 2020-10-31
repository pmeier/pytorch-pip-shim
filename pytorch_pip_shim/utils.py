import contextlib
import importlib
import optparse
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, cast
from unittest import mock

from . import computation_backend as cb

__all__ = [
    "InternalError",
    "canocialize_name",
    "apply_patch",
    "parse_pip_args",
    "computation_backend_options",
]


class InternalError(RuntimeError):
    def __init__(self) -> None:
        msg = (
            "Unexpected internal pytorch-pip-shim error. If you ever encounter this "
            "message during normal operation, please submit a bug report at "
            "https://github.com/pmeier/pytorch-pip-shim/issues"
        )
        super().__init__(msg)


def canocialize_name(name: str) -> str:
    return name.lower().replace("_", "-")


def parse_pip_args(args: List[str]) -> SimpleNamespace:
    if args[0] != "install":
        args = []

    parser = make_pip_args_parser()
    opts, _ = parser.parse_args(args)

    return SimpleNamespace(
        computation_backend=process_computation_backend(opts), nightly=opts.nightly
    )


# adapted from https://stackoverflow.com/a/9307174
class PassThroughOptionParser(optparse.OptionParser):
    def __init__(self, add_help_option: bool = False, **kwargs: Any) -> None:
        super().__init__(add_help_option=add_help_option, **kwargs)

    def _process_args(
        self, largs: List[str], rargs: List[str], values: optparse.Values
    ) -> None:
        while rargs:
            try:
                super()._process_args(largs, rargs, values)
            except (optparse.BadOptionError, optparse.AmbiguousOptionError) as error:
                largs.append(error.opt_str)


def make_pip_args_parser() -> PassThroughOptionParser:
    parser = PassThroughOptionParser()
    parser.add_option(
        "--pre", dest="nightly", action="store_true", default=False, help="nightly"
    )
    for option in computation_backend_options():
        parser.add_option(option)
    return parser


def computation_backend_options() -> Tuple[optparse.Option, ...]:
    return (
        optparse.Option(
            "--computation-backend",
            help=(
                "Computation backend for compiled PyTorch distributions, "
                "e.g. 'cu92', 'cu101', or 'cpu'. "
                "If not specified, the computation backend is detected from the "
                "available hardware, preferring CUDA over CPU."
            ),
        ),
        optparse.Option(
            "--cpu",
            action="store_true",
            default=False,
            help=(
                "Shortcut for '--computation-backend=cpu'. "
                "If '--computation-backend' is used simultaneously, "
                "it takes precedence over '--cpu'."
            ),
        ),
    )


def process_computation_backend(opts: optparse.Values) -> cb.ComputationBackend:
    if opts.computation_backend is not None:
        return cb.ComputationBackend.from_str(opts.computation_backend)

    if opts.cpu:
        return cb.CPUBackend()

    return cb.detect()


Args = Union[Tuple[()], Tuple[Any], Tuple[Any, ...]]
Kwargs = Dict[str, Any]


@contextlib.contextmanager
def apply_patch(
    target: str,
    preprocessing: Optional[Callable[[Args, Kwargs], Tuple[Args, Kwargs]]] = None,
    context: Optional[
        Callable[[Args, Kwargs], contextlib._GeneratorContextManager]
    ] = None,
    postprocessing: Optional[Callable[[Args, Kwargs, Any], Any]] = None,
) -> Iterator[None]:
    preprocessing = preprocessing or preprocessing_noop
    context = context or context_noop
    postprocessing = postprocessing or postprocessing_noop

    fn = import_fn(target)

    def new(*args: Any, **kwargs: Any) -> Any:
        preprocessing_ = cast(
            Callable[[Args, Kwargs], Tuple[Args, Kwargs]],
            preprocessing,
        )
        context_ = cast(
            Callable[[Args, Kwargs], contextlib._GeneratorContextManager],
            context,
        )
        postprocessing_ = cast(Callable[[Args, Kwargs, Any], Any], postprocessing)

        args, kwargs = preprocessing_(args, kwargs)
        with context_(args, kwargs):
            output = fn(*args, **kwargs)
        return postprocessing_(args, kwargs, output)

    with mock.patch(target, new=new):
        yield


def preprocessing_noop(args: Args, kwargs: Kwargs) -> Tuple[Args, Kwargs]:
    return args, kwargs


def postprocessing_noop(args: Args, kwargs: Kwargs, output: Any) -> Any:
    return output


@contextlib.contextmanager
def context_noop(args: Args, kwargs: Kwargs) -> Iterator[None]:
    yield


def import_fn(target: str) -> Callable:
    attrs = []
    name = target
    while name:
        try:
            module = importlib.import_module(name)
            break
        except ImportError:
            name, attr = name.rsplit(".", 1)
            attrs.append(attr)
    else:
        raise InternalError

    obj = module
    for attr in attrs[::-1]:
        obj = getattr(obj, attr)

    return cast(Callable, obj)
