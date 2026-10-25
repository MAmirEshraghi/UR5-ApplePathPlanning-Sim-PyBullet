"""
Vendored SB3 algorithms and helpers (originally the OSU pruning-sb3 project).

Checkpoints saved before the agent/ refactor pickle classes under the old top-level name
``apple_picking_sb3.<...>``. The alias below resolves those names to this package, so old
checkpoints still load with ``RecurrentPPOAEWithExpert.load``.
"""
import importlib
import importlib.abc
import importlib.util
import sys

_LEGACY_NAME = "apple_picking_sb3"


class _LegacyNameAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == _LEGACY_NAME or fullname.startswith(_LEGACY_NAME + "."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(__name__ + spec.name[len(_LEGACY_NAME):])

    def exec_module(self, module):
        pass


if not any(isinstance(finder, _LegacyNameAlias) for finder in sys.meta_path):
    sys.meta_path.append(_LegacyNameAlias())


def pruning_gym():
    return None
