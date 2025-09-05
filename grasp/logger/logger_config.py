import logging
import os
from typing import Protocol, Optional
from grasp.utils.constants import ROOT_DIR


class ExternalLoggerProtocol(Protocol):
    """Protocol defining the interface for external loggers."""

    @staticmethod
    def debug(msg: str) -> None: ...

    @staticmethod
    def info(msg: str) -> None: ...

    @staticmethod
    def error(msg: str) -> None: ...

    @staticmethod
    def warn(msg: str) -> None: ...

    @staticmethod
    def exception(msg: str) -> None: ...


class LoggerAdapter:
    """
    Adapter that provides a unified logging interface for GraSP.

    Supports both Python's standard logging and external loggers
    that follow the ExternalLoggerProtocol interface.
    """

    def __init__(self, external_logger: Optional[ExternalLoggerProtocol] = None):
        self._external_logger = external_logger
        self._internal_logger = None

        if not external_logger:
            self._setup_internal_logger()

    def _setup_internal_logger(self) -> None:
        """Setup the default GraSP internal logger."""
        log_dir = os.path.join(ROOT_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "out.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s default - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file),
            ],
        )
        self._internal_logger = logging.getLogger()

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log a debug message."""
        formatted_msg = msg % args if args else msg
        if self._external_logger:
            self._external_logger.debug(formatted_msg)
        else:
            self._internal_logger.debug(formatted_msg, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log an info message."""
        formatted_msg = msg % args if args else msg
        if self._external_logger:
            self._external_logger.info(formatted_msg)
        else:
            self._internal_logger.info(formatted_msg, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log a warning message."""
        formatted_msg = msg % args if args else msg
        if self._external_logger:
            self._external_logger.warn(formatted_msg)
        else:
            self._internal_logger.warning(formatted_msg, **kwargs)

    def warn(self, msg: str, *args, **kwargs) -> None:
        """Alias for warning() to maintain compatibility."""
        self.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log an error message."""
        formatted_msg = msg % args if args else msg
        if self._external_logger:
            self._external_logger.error(formatted_msg)
        else:
            self._internal_logger.error(formatted_msg, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log an error message with exception information."""
        formatted_msg = msg % args if args else msg
        if self._external_logger:
            self._external_logger.exception(formatted_msg)
        else:
            self._internal_logger.exception(formatted_msg, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log a critical message."""
        formatted_msg = msg % args if args else msg
        if self._external_logger:
            # External loggers might not have critical, fallback to error
            if hasattr(self._external_logger, "critical"):
                self._external_logger.critical(formatted_msg)
            else:
                self._external_logger.error(f"CRITICAL: {formatted_msg}")
        else:
            self._internal_logger.critical(formatted_msg, **kwargs)

    def set_external_logger(self, external_logger: ExternalLoggerProtocol) -> None:
        """
        Switch to using an external logger.

        Args:
            external_logger: External logger instance following ExternalLoggerProtocol
        """
        self._external_logger = external_logger

    def reset_to_internal(self) -> None:
        """Reset to using GraSP's internal logger."""
        self._external_logger = None
        if not self._internal_logger:
            self._setup_internal_logger()

    def is_using_external(self) -> bool:
        """Check if currently using an external logger."""
        return self._external_logger is not None


# Global logger adapter instance
_logger_adapter = LoggerAdapter()

# Expose the adapter as 'logger' for backward compatibility
logger = _logger_adapter


def set_external_logger(external_logger: ExternalLoggerProtocol) -> None:
    """
    Configure GraSP to use an external logger.

    Args:
        external_logger: External logger instance

    Example:
        >>> import CustomLogger
        >>> import grasp
        >>> grasp.logger.set_external_logger(NaginiLogger)
        >>> # Now all GraSP logs will use NaginiLogger
    """
    global _logger_adapter
    _logger_adapter.set_external_logger(external_logger)


def reset_to_internal_logger() -> None:
    """Reset GraSP to use its internal logger."""
    global _logger_adapter
    _logger_adapter.reset_to_internal()


def configure_logger(debug_mode: bool, clear_logs: bool, run_name: str) -> None:
    """
    Configure the internal logger (only affects internal logger, not external ones).

    Args:
        debug_mode: Enable debug mode
        clear_logs: Clear existing log files
        run_name: Run name for log formatting
    """
    global _logger_adapter

    # Only configure if using internal logger
    if not _logger_adapter.is_using_external():
        log_dir = os.path.join(ROOT_DIR, "logs")
        log_file = os.path.join(log_dir, "out.log")

        if clear_logs and os.path.exists(log_file):
            os.remove(log_file)

        # Create logs directory if not present
        if not os.path.exists("logs"):
            os.makedirs("logs")

        run_name_text = f" - {run_name}" if run_name else ""

        if debug_mode:
            openai_logger = logging.getLogger("openai")
            httpx_logger = logging.getLogger("httpx")
            openai_logger.setLevel(logging.DEBUG)
            httpx_logger.setLevel(logging.DEBUG)
            logging.basicConfig(
                level=logging.DEBUG,
                format=f"%(asctime)s - %(levelname)s{run_name_text} - %(message)s",
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(log_file),
                ],
                force=True,
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format=f"%(asctime)s - %(levelname)s{run_name_text} - %(message)s",
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(log_file),
                ],
                force=True,
            )

        _logger_adapter._internal_logger = logging.getLogger()


__all__ = [
    "logger",
    "set_external_logger",
    "reset_to_internal_logger",
    "configure_logger",
    "ExternalLoggerProtocol",
    "LoggerAdapter",
]
