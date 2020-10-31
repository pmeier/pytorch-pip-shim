import argparse
import sys
from typing import List, Optional

import pytorch_pip_shim

from ..utils import canocialize_name
from .commands import make_command

__all__ = ["main"]


def main(args: Optional[List[str]] = None) -> None:
    args = parse_args(args)
    cmd = make_command(args)
    cmd.run(args)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    if args is None:
        args = sys.argv[1:]

    parser = make_parser()
    return parser.parse_args(args)


def make_parser() -> argparse.ArgumentParser:
    prog = canocialize_name(pytorch_pip_shim.__name__)
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help=f"show {prog} version and path and exit",
    )

    subparsers = parser.add_subparsers(dest="subcommand", title="subcommands")
    add_insert_parser(subparsers)
    add_remove_parser(subparsers)
    add_status_parser(subparsers)
    add_detect_parser(subparsers)

    return parser


def add_file_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "file",
        type=str,
        nargs="?",
        help=(
            "pip main file. If not specified it is derived from pip associated with "
            "the running interpreter."
        ),
    )


SubParsers = argparse._SubParsersAction


def add_insert_parser(subparsers: SubParsers) -> None:
    parser = subparsers.add_parser("insert", description="Insert the shim")
    add_file_argument(parser)


def add_remove_parser(subparsers: SubParsers) -> None:
    parser = subparsers.add_parser(
        "remove",
        description="Remove the shim",
    )
    add_file_argument(parser)


def add_status_parser(subparsers: SubParsers) -> None:
    parser = subparsers.add_parser(
        "status",
        description="Status of the shim. If inserted returns 0 otherwise 1.",
    )
    add_file_argument(parser)
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Do not print the status to STDOUT.",
    )


def add_detect_parser(subparsers: SubParsers) -> None:
    subparsers.add_parser(
        "detect",
        description=(
            "Detect the computation backend from the available hardware, "
            "preferring CUDA over CPU."
        ),
    )
