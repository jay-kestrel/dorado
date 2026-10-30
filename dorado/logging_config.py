# -*- coding: utf-8 -*-
"""
Logging configuration for the dorado package.

Provides centralized logging setup and backward-compatible handling
of the deprecated `verbose` parameter.

Project Homepage: https://github.com/passaH2O/dorado
"""
import logging
import os
import warnings

# Package-level logger
logger = logging.getLogger("dorado")

# Environment variable controlling default logger visibility
ENV_LEVEL_VAR = "DORADO_LOG_LEVEL"


def configure_from_env():
    """Configure the 'dorado' logger from the DORADO_LOG_LEVEL env var.

    Called at package import. Behavior:

    - Unset: no handler attached and no level change. The logger keeps
      Python's default effective level (WARNING via root) and
      ``logging.lastResort`` still surfaces WARNING-and-above records to
      stderr, preserving out-of-the-box behavior.
    - Valid value: the logger level is set to that value and one
      StreamHandler (stderr) with the standard formatter is attached if
      none is present.
    - Invalid value: a warning is logged naming the bad value, then
      WARNING is used. The import never fails.
    """
    level = os.environ.get(ENV_LEVEL_VAR)
    if not level:
        return
    try:
        numeric = getattr(logging, level.strip().upper())
    except (AttributeError, ValueError):
        logger.warning(
            "Invalid %s=%r - falling back to WARNING", ENV_LEVEL_VAR, level
        )
        numeric = logging.WARNING
    logger.setLevel(numeric)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(h)


configure_from_env()


def setup_logging(level="WARNING", handler=None):
    """Configure dorado logging.

    **Inputs** :

        level : `str`, optional
            Logging level. One of "DEBUG", "INFO", "WARNING", "ERROR",
            "CRITICAL". Default is "WARNING".

        handler : `logging.Handler`, optional
            Custom logging handler. If not provided, a StreamHandler
            writing to stderr is used.

    **Example**::

        import dorado

        # Show informational messages
        dorado.setup_logging("INFO")

        # Log to a file
        import logging
        dorado.setup_logging(
            "DEBUG", handler=logging.FileHandler("dorado.log")
        )

    """
    try:
        numeric_level = getattr(logging, level.upper())
    except AttributeError:
        raise ValueError(
            f"Invalid logging level: {level}. Must be one of "
            "DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )
    logger.setLevel(numeric_level)
    if not logger.handlers:
        h = handler or logging.StreamHandler()
        h.setFormatter(
            logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(h)


def handle_verbose_deprecation(verbose):
    """Map deprecated verbose parameter to logging behavior.

    This function provides backward compatibility for the deprecated
    `verbose` parameter. It is a non-persistent shim: unlike
    :func:`setup_logging`, a call never permanently reconfigures the
    global ``dorado`` logger. Passing True (or None, which the API
    docstrings document as equivalent to True) raises the logger to
    INFO so legacy verbose runs still show detail. Passing False no
    longer forces the logger to ERROR — the currently configured level
    is left in place, so warnings remain visible after the call. Users
    who want a quieter logger should call :func:`setup_logging`
    directly.

    **Inputs** :

        verbose : `bool` or `None`
            Legacy verbose parameter. If True or None, sets logging to
            INFO level. If False, leaves the currently configured
            level (WARNING by default) untouched.

    """
    if verbose is None:
        # docstrings promise None is equivalent to True (INFO level)
        verbose = True
    warnings.warn(
        "The 'verbose' parameter is deprecated. "
        "Use dorado.setup_logging() instead.",
        DeprecationWarning,
        stacklevel=3
    )
    if verbose:
        setup_logging("INFO")
