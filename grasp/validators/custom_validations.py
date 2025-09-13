from typing import Any

from grasp.logger.logger_config import logger


def validate_is_greater_than(self, field_name: str, other: Any) -> bool:
    """Custom validation for 'is_greater_than' rule"""
    try:
        if self <= other:
            logger.error(
                f"Validation failed for field '{field_name}': Value should be greater than {other}. Got {self}."
            )
            return False
        else:
            logger.info(f"is_greater_than validation passed for field '{field_name}'.")
        return True
    except Exception as e:
        logger.error(f"Error during 'is_greater_than' validation for field '{field_name}': {e}")
        return False


def validate_is_equal_to(self, field_name: str, other: Any) -> bool:
    """Custom validation for 'is_equal_to' rule"""
    try:
        if self != other:
            logger.error(
                f"Validation failed for field '{field_name}': Value should be equal to {other}. Got {self}."
            )
            return False
        else:
            logger.info(f"is_equal_to validation passed for field '{field_name}'.")
        return True
    except Exception as e:
        logger.error(f"Error during 'is_equal_to' validation for field '{field_name}': {e}")
        return False


def validate_is_less_than(self, field_name: str, other: Any) -> bool:
    """Custom validation for 'is_less_than' rule"""
    try:
        if self >= other:
            logger.error(
                f"Validation failed for field '{field_name}': Value should be less than {other}. Got {self}."
            )
            return False
        else:
            logger.info(f"is_less_than validation passed for field '{field_name}'.")
        return True
    except Exception as e:
        logger.error(f"Error during 'is_less_than' validation for field '{field_name}': {e}")
        return False
