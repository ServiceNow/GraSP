from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field, create_model

from grasp.logger.logger_config import logger


class SchemaConfigParser:
    """Dynamic schema parsing class"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.schema_type: Optional[str] = None  # Will be 'class' or 'schema'
        self.schema_data: Optional[dict[Any, Any]] = None
        self.class_path: Optional[str] = None

        self._parse_schema()

    def _parse_schema(self):
        """Parse and validate the schema configuration."""
        schema = self.config.get("schema")

        if schema is None:
            raise ValueError("Schema field is required in structured_output configuration")

        if isinstance(schema, str):
            # It's a class path
            self._handle_class_path(schema)
        elif isinstance(schema, dict):
            # It's a schema definition
            self._handle_schema_dict(schema)
        else:
            raise ValueError(
                f"Schema must be either a string (class path) or dictionary (schema definition), "
                f"got {type(schema).__name__}"
            )

    def _handle_class_path(self, class_path: str):
        """Handle class path validation and loading."""
        if not self._is_valid_class_path(class_path):
            raise ValueError(f"Invalid class path format: {class_path}")

        # Validate that the class can be imported
        try:
            self._import_class(class_path)
        except Exception as e:
            raise ValueError(f"Failed to import class '{class_path}': {str(e)}")

        self.schema_type = "class"
        self.class_path = class_path
        self.schema_data = None

    def _handle_schema_dict(self, schema_dict: dict):
        """Handle schema dictionary validation."""
        # Validate the schema structure
        if not self._is_valid_schema_dict(schema_dict):
            raise ValueError("Invalid schema dictionary structure")

        self.schema_type = "schema"
        self.schema_data = schema_dict
        self.class_path = None

    def _is_valid_class_path(self, class_path: str) -> bool:
        """Validate class path format."""
        if not isinstance(class_path, str) or not class_path.strip():
            return False

        # Basic validation: should contain at least one dot and valid Python identifiers
        parts = class_path.split(".")
        if len(parts) < 2:
            return False

        # Check if all parts are valid Python identifiers
        return all(part.isidentifier() for part in parts)

    def _import_class(self, class_path: str):
        """Import and return the class from the given path."""
        module_path, class_name = class_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            # Verify it's actually a class
            if not inspect.isclass(cls):
                raise ValueError(f"'{class_path}' is not a class")

            return cls
        except ImportError as e:
            raise ImportError(f"Cannot import module '{module_path}': {str(e)}")
        except AttributeError:
            raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'")

    def _is_valid_schema_dict(self, schema_dict: dict) -> bool:
        """Validate schema dictionary structure."""
        # Check if it has the expected structure with 'fields'
        if not isinstance(schema_dict, dict):
            return False

        if "fields" not in schema_dict:
            raise ValueError("Schema dictionary must contain 'fields' key")

        fields = schema_dict["fields"]
        if not isinstance(fields, dict):
            raise ValueError("'fields' must be a dictionary")

        # Validate each field
        for field_name, field_config in fields.items():
            if not isinstance(field_config, dict):
                raise ValueError(f"Field '{field_name}' configuration must be a dictionary")

            if "type" not in field_config:
                raise ValueError(f"Field '{field_name}' must have a 'type' specified")

            # Optional: validate field types
            valid_types = {"str", "int", "float", "bool", "list", "dict"}
            field_type = field_config["type"]
            if field_type not in valid_types:
                raise ValueError(
                    f"Invalid type '{field_type}' for field '{field_name}'. "
                    f"Valid types: {', '.join(valid_types)}"
                )

        return True


class StructuredOutputConfig:
    """Configuration class for structured output"""

    def __init__(self, config: dict[str, Any], key_present: bool = True):
        self.enabled = config.get("enabled", key_present)  # True if key present, False if absent
        if self.enabled:
            # Use the unified parser
            self.parser = SchemaConfigParser(config)
            self.type = self.parser.schema_type
            self.schema = self.parser.schema_data
            self.class_path = self.parser.class_path
        self.fallback_strategy = config.get(
            "fallback_strategy", "instruction"
        )  # instruction, post_process
        self.retry_on_parse_error = config.get("retry_on_parse_error", True)
        self.max_parse_retries = config.get("max_parse_retries", 2)

    def get_pydantic_model(self) -> Optional[Type[BaseModel]]:
        """Get or create pydantic model based on configuration"""
        if not self.enabled:
            return None

        if self.type == "class" and self.class_path:
            return self._load_class_from_path(self.class_path)
        elif self.type == "schema" and self.schema:
            return self._create_pydantic_from_yaml(self.schema)

        return None

    def _load_class_from_path(self, class_path: str) -> Type[BaseModel]:
        """Load pydantic class from module path"""
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            pydantic_class: Type[BaseModel] = getattr(module, class_name)

            if not issubclass(pydantic_class, BaseModel):
                raise ValueError(f"Class {class_name} is not a Pydantic BaseModel")

            return pydantic_class
        except Exception as e:
            logger.error(f"Failed to load class from path {class_path}: {e}")
            raise

    def _create_pydantic_from_yaml(self, schema_dict: dict[str, Any]) -> Type[BaseModel]:
        """Create pydantic model from YAML schema definition"""
        try:
            field_definitions: Dict[str, Any] = {}
            for field_name, field_config in schema_dict.get("fields", {}).items():
                field_type = self._get_python_type(field_config.get("type", "str"))
                default = field_config.get("default", ...)
                description = field_config.get("description", "")
                # Use Field to attach metadata
                field_definitions[field_name] = (
                    field_type,
                    Field(default=default, description=description),
                )

            model_name = schema_dict.get("name", "DynamicModel")
            model: Type[BaseModel] = create_model(model_name, **field_definitions)
            return model
        except Exception as e:
            logger.error(f"Failed to create pydantic model from YAML schema: {e}")
            raise

    def _get_python_type(self, type_str: str) -> Type:
        """Convert string type to Python type"""
        type_mapping = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "number": float,
            "bool": bool,
            "boolean": bool,
            "list": list,
            "array": list,
            "dict": dict,
            "object": dict,
        }
        return type_mapping.get(type_str.lower(), str)
