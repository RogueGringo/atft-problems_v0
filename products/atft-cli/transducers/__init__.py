"""
transducers/__init__.py — Auto-discovery registry for atft transducers.

Any .py module dropped into this directory that defines a BaseTransducer
subclass (with a non-"base" `name`) is automatically available via
`get_transducer` and `list_transducers`.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path

from .base import BaseTransducer

_REGISTRY: dict[str, type[BaseTransducer]] | None = None


def _build_registry() -> dict[str, type[BaseTransducer]]:
    registry: dict[str, type[BaseTransducer]] = {}
    package_dir = str(Path(__file__).parent)
    package_name = __name__  # "transducers"

    for module_info in pkgutil.iter_modules([package_dir]):
        mod_name = module_info.name
        if mod_name in ("base", "__init__"):
            continue
        full_name = f"{package_name}.{mod_name}"
        try:
            module = importlib.import_module(full_name)
        except Exception:
            continue
        for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseTransducer)
                and obj is not BaseTransducer
                and obj.name != "base"
            ):
                registry[obj.name] = obj
    return registry


def _get_registry() -> dict[str, type[BaseTransducer]]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_transducer(name: str) -> BaseTransducer:
    """Instantiate a transducer by name. Raises KeyError if unknown."""
    registry = _get_registry()
    if name not in registry:
        available = list(registry.keys())
        raise KeyError(
            f"Unknown transducer '{name}'. Available: {available}"
        )
    return registry[name]()


def list_transducers() -> list[str]:
    """Return a sorted list of all registered transducer names."""
    return sorted(_get_registry().keys())
