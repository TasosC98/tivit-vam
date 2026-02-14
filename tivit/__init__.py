"""TiViT-Piano top-level package shim + compatibility aliases.

This repo is a monorepo-style layout:
- `theory/` lives at:      <monorepo_root>/theory
- training code lives at:  <monorepo_root>/tivit  (this directory)
and many modules historically import `tivit.core.*`, `tivit.data.*`, etc.

This shim ensures:
1) `import tivit.theory` works (by adding monorepo root to sys.path)
2) `import tivit.core`, `tivit.data`, `tivit.pipelines`, ... resolve to the
   top-level folders (core/, data/, pipelines/, ...).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

# Ensure monorepo root is on sys.path so `import theory` works.
# Current file: <monorepo_root>/tivit/tivit/__init__.py
# parents[2] -> <monorepo_root>
_MONOREPO_ROOT = Path(__file__).resolve().parents[2]
if str(_MONOREPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MONOREPO_ROOT))

# Also ensure <monorepo_root>/tivit is on sys.path (often already is, but safe)
_TIVIT_ROOT = Path(__file__).resolve().parent.parent
if str(_TIVIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_TIVIT_ROOT))


def _import_theory() -> ModuleType | None:
    """Load the underlying `theory` package if available (optional)."""
    try:
        module = importlib.import_module("theory")
    except ModuleNotFoundError:
        return None
    sys.modules.setdefault(__name__ + ".theory", module)
    return module


_THEORY = _import_theory()

# Compatibility aliases: map `tivit.<name>` -> `<name>` (flat-layout folders)
_ALIASES = [
    "core",
    "data",
    "pipelines",
    "models",
    "train",
    "utils",
    "losses",
    "metrics",
    "preproc",
    "postproc",
    "decoder",
    "priors",
    "calibration",
]

for _name in _ALIASES:
    try:
        _mod = importlib.import_module(_name)
        sys.modules.setdefault(__name__ + f".{_name}", _mod)
    except Exception:
        pass

# Re-export theory helpers when theory exists
if TYPE_CHECKING:
    from theory.key_prior import KeyAwarePrior as KeyAwarePrior
    from theory.key_prior import KeyPriorConfig as KeyPriorConfig
    from theory.key_prior import build_key_profiles as build_key_profiles
    from theory.key_prior_runtime import (
        KeyPriorRuntimeSettings as KeyPriorRuntimeSettings,
        resolve_key_prior_settings as resolve_key_prior_settings,
        apply_key_prior_to_logits as apply_key_prior_to_logits,
    )
else:
    if _THEORY is not None:
        theory = _THEORY
        KeyAwarePrior = _THEORY.KeyAwarePrior
        KeyPriorConfig = _THEORY.KeyPriorConfig
        build_key_profiles = _THEORY.build_key_profiles
        KeyPriorRuntimeSettings = _THEORY.KeyPriorRuntimeSettings
        resolve_key_prior_settings = _THEORY.resolve_key_prior_settings
        apply_key_prior_to_logits = _THEORY.apply_key_prior_to_logits
    else:
        theory = None  # type: ignore[assignment]

__all__ = [
    "theory",
    "KeyAwarePrior",
    "KeyPriorConfig",
    "build_key_profiles",
    "KeyPriorRuntimeSettings",
    "resolve_key_prior_settings",
    "apply_key_prior_to_logits",
]


def __getattr__(name: str) -> object:
    if _THEORY is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}' (theory not available)")
    return getattr(_THEORY, name)


def __dir__() -> list[str]:
    if _THEORY is None:
        return sorted(set(__all__))
    return sorted(set(__all__) | set(dir(_THEORY)))
