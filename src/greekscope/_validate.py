"""Input validation helpers shared by all greekscope modules."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["check_kind", "check_inputs"]


def check_kind(kind: str) -> str:
    """Normalise and validate an option kind.

    Parameters
    ----------
    kind : str
        ``"call"`` or ``"put"`` (case-insensitive, surrounding whitespace ignored).

    Returns
    -------
    str
        Lower-case ``"call"`` or ``"put"``.

    Raises
    ------
    ValueError
        If ``kind`` is not a string or is not one of the two allowed values.
    """
    if not isinstance(kind, str):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    k = kind.strip().lower()
    if k not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return k


def _as_float_array(x: Any, name: str) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {x!r}") from exc
    return a


def check_inputs(
    S: Any,
    K: Any,
    T: Any,
    sigma: Any,
    *,
    r: Any = None,
    q: Any = None,
) -> None:
    """Validate the common option inputs (scalars or arrays).

    Parameters
    ----------
    S, K : float or array_like
        Spot and strike; must be finite and strictly positive.
    T : float or array_like
        Time to expiry in years; must be finite and ``>= 0``.
    sigma : float or array_like
        Volatility; must be finite and ``>= 0``.
    r, q : float or array_like, optional
        If given, must be finite (any sign allowed).

    Raises
    ------
    ValueError
        With a message naming the offending argument.
    """
    S_ = _as_float_array(S, "S")
    K_ = _as_float_array(K, "K")
    T_ = _as_float_array(T, "T")
    sg = _as_float_array(sigma, "sigma")
    if not np.all(np.isfinite(S_)) or not np.all(S_ > 0):
        raise ValueError(f"S (spot) must be finite and > 0, got {S!r}")
    if not np.all(np.isfinite(K_)) or not np.all(K_ > 0):
        raise ValueError(f"K (strike) must be finite and > 0, got {K!r}")
    if not np.all(np.isfinite(T_)) or not np.all(T_ >= 0):
        raise ValueError(f"T (time to expiry) must be finite and >= 0, got {T!r}")
    if not np.all(np.isfinite(sg)) or not np.all(sg >= 0):
        raise ValueError(f"sigma (volatility) must be finite and >= 0, got {sigma!r}")
    for name, val in (("r", r), ("q", q)):
        if val is not None and not np.all(np.isfinite(_as_float_array(val, name))):
            raise ValueError(f"{name} must be finite, got {val!r}")


def _scalar_or_array(x: Any) -> Any:
    """Return a Python float for 0-d input, otherwise the ndarray unchanged."""
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    return arr
