from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pydorado")
except PackageNotFoundError:
    __version__ = "unknown"


from . import lagrangian_walker
from . import parallel_routing
from . import particle_track
from . import routines
from . import spatial
from .logging_config import setup_logging, logger

# Optional native routing kernel (Rust/PyO3). The extension ships in
# maturin-built wheels but is absent in pure-Python checkouts and builds, so
# keep `import dorado` working everywhere (spec D3 wires the backend switch).
try:
    from . import _native
except ImportError:  # pragma: no cover - depends on wheel contents
    _native = None
