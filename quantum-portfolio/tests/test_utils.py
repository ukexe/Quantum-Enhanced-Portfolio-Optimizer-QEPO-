"""Tests for utils module."""

import json
import logging

import pytest

from qepo import utils


class TestJsonFormatter:
    """Test JsonFormatter class."""

    def test_json_formatter_format(self):
        """Test JSON formatter creates valid JSON."""
        formatter = utils.JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["logger"] == "test_logger"

    def test_json_formatter_with_args(self):
        """Test JSON formatter with message arguments."""
        formatter = utils.JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error: %s",
            args=("test error",),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "ERROR"
        assert parsed["message"] == "Error: test error"
        assert parsed["logger"] == "test_logger"


class TestConfigureLogging:
    """Test configure_logging function."""

    def test_configure_logging_default_level(self):
        """Test configure_logging with default level."""
        utils.configure_logging()

        root = logging.getLogger()
        assert root.level == logging.INFO
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, utils.JsonFormatter)

    def test_configure_logging_custom_level(self):
        """Test configure_logging with custom level."""
        utils.configure_logging(level=logging.DEBUG)

        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, utils.JsonFormatter)

    def test_configure_logging_clears_handlers(self):
        """Test configure_logging clears existing handlers."""
        # Add a handler first
        root = logging.getLogger()
        old_handler = logging.StreamHandler()
        root.addHandler(old_handler)

        # Configure logging
        utils.configure_logging()

        # Should have only one handler now
        assert len(root.handlers) == 1
        assert root.handlers[0] != old_handler
