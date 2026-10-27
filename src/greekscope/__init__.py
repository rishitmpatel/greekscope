"""greekscope: option pricing, Greeks, implied volatility and strategy analysis."""

from .bsm import d1, d2
from .bsm import price as bsm_price
from .greeks import all_greeks, delta, gamma, rho, theta, vanna, vega, volga
from .implied_vol import implied_vol
from .binomial import crr_price
from .montecarlo import MCResult, asian_mc_price, mc_price
from .heston import heston_price

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "d1",
    "d2",
    "bsm_price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "vanna",
    "volga",
    "all_greeks",
    "implied_vol",
    "crr_price",
    "MCResult",
    "mc_price",
    "asian_mc_price",
    "heston_price",
]

# Modules owned by another contributor; the package must import cleanly without them.
try:
    from . import strategy as _strategy
    from .strategy import *  # noqa: F401,F403

    __all__ += [n for n in getattr(_strategy, "__all__", []) if n not in __all__]
except ImportError:
    pass

try:
    from . import surface as _surface
    from .surface import *  # noqa: F401,F403

    __all__ += [n for n in getattr(_surface, "__all__", []) if n not in __all__]
except ImportError:
    pass
