import importlib
from typing import Any, cast

from pydantic import BaseModel

from grasp.validators.type_parser import ParsedType, TypeParser


def parse_type_string(type_str: str) -> ParsedType:
    parser = TypeParser()
    return parser.parse(type_str)


def evaluate_type(type_str: str) -> ParsedType:
    """
    Evaluates a complex type annotation string and returns the corresponding Python type.
    """
    try:
        return parse_type_string(type_str)
    except ValueError as e:
        raise ValueError(f"Error evaluating type: {str(e)}")


def process_custom_fields(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Processes the custom fields from YAML, handling nested types like list[list[str]].
    """
    """
       Processes fields from the YAML config, extracting name, type, and additional rules.
       """
    result_fields: list[dict[str, Any]] = []
    raw_fields = config.get("fields", [])
    if not isinstance(raw_fields, list):
        raise TypeError(f"Expected 'fields' to be a list of dicts, got {type(raw_fields).__name__}")
    for field in raw_fields:
        if not isinstance(field, dict):
            raise TypeError(
                f"Each item in 'fields' must be a dict, got {type(field).__name__}: {field}"
            )
        try:
            # Initialize the field information dictionary
            field_info = {
                "name": field["name"],
                "type": evaluate_type(str(field["type"])),
            }

            # Add additional keys (e.g., is_greater_than, is_not_empty, etc.) dynamically
            for key, value in field.items():
                if key != "name" and key != "type":
                    field_info[key] = value

            result_fields.append(field_info)

        except KeyError as e:
            raise KeyError(f"KeyError: Missing expected key {e} in field definition: {field}")
        except TypeError as e:
            raise TypeError(
                f"TypeError: Invalid data type encountered while processing field: {field}. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error occurred while processing field: {field}. Error: {e}"
            )

    return result_fields


def resolve_schema_class(schema_path: str) -> type[BaseModel]:
    """
    Resolves the schema class from the path.
    Example: "validators.custom_schema.CustomUserSchema" -> CustomUserSchema class.
    """
    try:
        # Split the schema path into module path and class name
        module_path, class_name = schema_path.rsplit(".", 1)
        # Dynamically import the module using importlib
        module = importlib.import_module(module_path)
        # Get the class from the imported module
        attr = getattr(module, class_name)
        if not isinstance(attr, type):
            raise ValueError(f"{class_name} resolved to non-type: {type(attr).__name__}")
        schema_class = cast(type[BaseModel], attr)
        if not issubclass(schema_class, BaseModel):
            raise ValueError(f"{schema_class} is not a subclass of pydantic.BaseModel")
        return schema_class
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Invalid schema path: {schema_path}") from e
