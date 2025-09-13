import functools
import warnings


def future_deprecation(reason=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"'{func.__name__}' is scheduled for future deprecation"
            if reason:
                msg += f": {reason}"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        setattr(wrapper, "_future_deprecation", True)  # type: ignore[attr-defined]
        return wrapper

    return decorator
