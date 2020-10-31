import argparse
import sys
from abc import ABC, abstractmethod
from os import path
from typing import Dict, NoReturn, Optional, Type

import pytorch_pip_shim

from .. import shim
from ..computation_backend import detect
from ..utils import canocialize_name

__all__ = ["make_command"]


class Command(ABC):
    def run(self, args: argparse.Namespace) -> NoReturn:
        ok = self._run(args)
        sys.exit(int(not ok) if ok is not None else 0)

    @abstractmethod
    def _run(self, args: argparse.Namespace) -> Optional[bool]:
        pass


class GlobalCommand(Command):
    def _run(self, args: argparse.Namespace) -> bool:
        if args.version:
            name = canocialize_name(pytorch_pip_shim.__name__)
            version = pytorch_pip_shim.__version__
            root = path.abspath(path.join(path.dirname(__file__), ".."))
            print(f"{name}=={version} from {root}")
            return True

        return False


class InsertCommand(Command):
    def _run(self, args: argparse.Namespace) -> None:
        shim.insert(args.file)


class RemoveCommand(Command):
    def _run(self, args: argparse.Namespace) -> None:
        shim.remove(args.file)


class StatusCommand(Command):
    def _run(self, args: argparse.Namespace) -> bool:
        is_inserted = shim.is_inserted(args.file)
        if not args.quiet:
            print(f"The shim {'is' if is_inserted else 'is NOT'} inserted.")
        return is_inserted


class DetectCommand(Command):
    def _run(self, args: argparse.Namespace) -> None:
        print(detect())


COMMAD_CLASSES: Dict[Optional[str], Type[Command]] = {
    None: GlobalCommand,
    "insert": InsertCommand,
    "remove": RemoveCommand,
    "status": StatusCommand,
    "detect": DetectCommand,
}


def make_command(args: argparse.Namespace) -> Command:
    return COMMAD_CLASSES[args.subcommand]()
