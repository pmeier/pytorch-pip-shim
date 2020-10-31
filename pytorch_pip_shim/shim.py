from os import path
from typing import Optional

import pip

__all__ = ["insert", "remove"]


IDENTIFIER = "# -*- pytorch-pip-shim -*-"
SHIM = """
try:
    import pytorch_pip_shim
except:
    import functools
    import types

    def patch(pip_main):
        @functools.wraps(pip_main)
        def shim(*args, **kwargs):
            return pip_main(*args, **kwargs)

        return shim

    pytorch_pip_shim = types.SimpleNamespace(patch=patch)

@pytorch_pip_shim.patch
"""

FILE = path.abspath(
    path.join(path.dirname(pip.__file__), "_internal", "cli", "main.py")
)


def is_inserted(file: Optional[str] = None) -> bool:
    if file is None:
        file = FILE

    with open(file, "r") as fh:
        return IDENTIFIER in fh.read()


def insert(file: Optional[str] = None) -> None:
    if file is None:
        file = FILE

    if is_inserted(file):
        return

    with open(file, "r") as fh:
        content = fh.readlines()

    idx = content.index(next(line for line in content if line.startswith("def main(")))
    content = (
        content[:idx] + [IDENTIFIER] + SHIM.splitlines(keepends=True) + content[idx:]
    )
    with open(file, "w") as fh:
        fh.writelines(content)


def remove(file: Optional[str] = None) -> None:
    if file is None:
        file = FILE

    if not is_inserted(file):
        return

    with open(file, "r") as fh:
        content = fh.readlines()

    lines = (line for line in content if line.startswith((IDENTIFIER, "def main(")))
    start_idx = content.index(next(lines))
    end_idx = content.index(next(lines))
    del content[start_idx:end_idx]

    with open(file, "w") as fh:
        fh.writelines(content)
