import importlib
import logging

import pytest

from dorado import logging_config

ENV_LEVEL_VAR = "DORADO_LOG_LEVEL"


@pytest.fixture
def clean_logger():
    """Reset the dorado logger to its pristine import state around a test.

    Earlier tests in this module (and the package import itself) mutate
    the module-level logger's level and handlers, so snapshot/restore of
    the live state is not enough — reset to the shipped defaults both
    before and after.
    """
    lg = logging_config.logger
    lg.setLevel(logging.NOTSET)
    lg.handlers = []
    yield lg
    lg.setLevel(logging.NOTSET)
    lg.handlers = []


@pytest.fixture(autouse=True)
def restore_dorado_logger():
    """Save and restore the global dorado logger around each test."""
    logger = logging_config.logger
    level = logger.level
    handlers = list(logger.handlers)
    yield
    logger.setLevel(level)
    logger.handlers[:] = handlers


def test_setup_logging():
    # Test default
    logging_config.setup_logging()
    assert logging_config.logger.level == logging.WARNING

    # Test setting level
    logging_config.setup_logging("INFO")
    assert logging_config.logger.level == logging.INFO

    # Test invalid level
    with pytest.raises(ValueError, match="Invalid logging level"):
        logging_config.setup_logging("INVALID")


def test_handle_verbose_deprecation():
    # verbose=True raises the logger to INFO
    with pytest.warns(DeprecationWarning, match="The 'verbose' parameter is deprecated"):
        logging_config.handle_verbose_deprecation(True)
    assert logging_config.logger.level == logging.INFO

    # verbose=False must not silence the logger: the pre-existing level
    # is kept, so warnings emitted afterwards stay visible
    logging_config.logger.setLevel(logging.WARNING)
    with pytest.warns(DeprecationWarning, match="The 'verbose' parameter is deprecated"):
        logging_config.handle_verbose_deprecation(False)
    assert logging_config.logger.isEnabledFor(logging.WARNING)

    # verbose=None is documented as equivalent to True (INFO level)
    with pytest.warns(DeprecationWarning, match="The 'verbose' parameter is deprecated"):
        logging_config.handle_verbose_deprecation(None)
    assert logging_config.logger.level == logging.INFO


def test_warning_visible_after_verbose_false(caplog):
    # a WARNING emitted after a verbose=False call is still visible
    logging_config.logger.setLevel(logging.WARNING)
    with pytest.warns(DeprecationWarning, match="The 'verbose' parameter is deprecated"):
        logging_config.handle_verbose_deprecation(False)
    logger = logging.getLogger("dorado")
    with caplog.at_level(logging.WARNING, logger="dorado"):
        logger.warning("still visible after verbose=False")
    assert "still visible after verbose=False" in caplog.text


def test_env_var_unset(monkeypatch, clean_logger):
    # Unset -> no handler attached and no level override: the logger keeps
    # its default effective level (WARNING via root), lastResort still
    # surfaces WARNING+ records.
    monkeypatch.delenv(ENV_LEVEL_VAR, raising=False)
    importlib.reload(logging_config)
    assert logging_config.logger.level == logging.NOTSET
    assert logging_config.logger.handlers == []


def test_env_var_valid(monkeypatch, clean_logger):
    # Valid value -> level applied and exactly one stderr StreamHandler
    # with the standard formatter attached.
    monkeypatch.setenv(ENV_LEVEL_VAR, "info")
    importlib.reload(logging_config)
    assert logging_config.logger.level == logging.INFO
    assert len(logging_config.logger.handlers) == 1
    handler = logging_config.logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.formatter._fmt == "%(name)s - %(levelname)s - %(message)s"

    # Re-calling configure_from_env() re-reads the variable and must not
    # stack up duplicate handlers.
    monkeypatch.setenv(ENV_LEVEL_VAR, "DEBUG")
    logging_config.configure_from_env()
    assert logging_config.logger.level == logging.DEBUG
    assert len(logging_config.logger.handlers) == 1


def test_env_var_invalid(monkeypatch, clean_logger, caplog):
    # Invalid value -> warning record naming the bad value, WARNING
    # fallback, and the import (reload) never crashes.
    monkeypatch.setenv(ENV_LEVEL_VAR, "NOT_A_LEVEL")
    # Asserts stay inside the context: caplog.at_level restores the
    # logger's previous level on exit, which would undo the reload's
    # setLevel(WARNING).
    with caplog.at_level(logging.WARNING, logger="dorado"):
        importlib.reload(logging_config)
        assert logging_config.logger.level == logging.WARNING
        matching = [
            r for r in caplog.records
            if "DORADO_LOG_LEVEL" in r.getMessage()
            and "NOT_A_LEVEL" in r.getMessage()
        ]
        assert matching, "expected a warning record naming the invalid value"
        assert matching[0].levelno == logging.WARNING


def test_configure_from_env_re_reads_variable(monkeypatch, clean_logger):
    # Without reload: unset is a no-op, then a direct call picks up the
    # newly set value.
    monkeypatch.delenv(ENV_LEVEL_VAR, raising=False)
    logging_config.configure_from_env()
    assert logging_config.logger.level == logging.NOTSET
    assert logging_config.logger.handlers == []

    monkeypatch.setenv(ENV_LEVEL_VAR, "ERROR")
    logging_config.configure_from_env()
    assert logging_config.logger.level == logging.ERROR
    assert len(logging_config.logger.handlers) == 1
