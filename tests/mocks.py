import builtins
import contextlib
import sys
import unittest.mock

__all__ = [
    "make_target",
    "patch_imports",
]


def make_target(*args, pkg="pytorch_pip_shim"):
    return ".".join((pkg, *args))


@contextlib.contextmanager
def patch_imports(*names, retain_condition=None, import_error_condition=None):
    if retain_condition is None:

        def retain_condition(name):
            return not any(name.startswith(name_) for name_ in names)

    if import_error_condition is None:

        def import_error_condition(name, globals, locals, fromlist, level):
            direct = name in names
            indirect = fromlist is not None and any(
                from_ in names for from_ in fromlist
            )
            return direct or indirect

    __import__ = builtins.__import__

    def patched_import(name, globals, locals, fromlist, level):
        if import_error_condition(name, globals, locals, fromlist, level):
            raise ImportError

        return __import__(name, globals, locals, fromlist, level)

    values = {
        name: module for name, module in sys.modules.items() if retain_condition(name)
    }

    with unittest.mock.patch.object(
        builtins, "__import__", new=patched_import
    ), unittest.mock.patch.dict(
        sys.modules,
        clear=True,
        values=values,
    ):
        yield
