"""GraSP Library Exceptions"""


class GraSPError(Exception):
    """Base exception for GraSP library."""

    pass


class ValidationError(GraSPError):
    """Validation error."""

    pass


class ExecutionError(GraSPError):
    """Execution error."""

    pass


class ConfigurationError(GraSPError):
    """Configuration error."""

    pass


class NodeError(GraSPError):
    """Node processing error."""

    pass


class DataError(GraSPError):
    """Data handling error."""

    pass


class ModelError(GraSPError):
    """Model error."""

    pass


class TimeoutError(GraSPError):
    """Timeout error."""

    pass


__all__ = [
    "GraSPError",
    "ValidationError",
    "ExecutionError",
    "ConfigurationError",
    "NodeError",
    "DataError",
    "ModelError",
    "TimeoutError",
]
