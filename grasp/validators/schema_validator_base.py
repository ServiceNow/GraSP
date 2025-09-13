from typing import Any, Optional, Type

from pydantic import BaseModel, TypeAdapter, ValidationError

from grasp.core.graph.graph_config import GraphConfig
from grasp.logger.logger_config import logger
from grasp.validators import custom_validations as custom_validations
from grasp.validators.yaml_loader import process_custom_fields, resolve_schema_class


class SchemaValidator:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        This method is used to create a single instance of SchemaValidator.
        If an instance already exists, it returns the same instance.
        """
        if cls._instance is None:
            # Create the instance and store it in _instance
            cls._instance = super(SchemaValidator, cls).__new__(cls)
        return cls._instance

    def __init__(self, graph_config: GraphConfig):
        """
        Initializes the SchemaValidator with the given GraphConfig Object.
        """
        # Avoid reinitializing the object if it's already been initialized
        if hasattr(self, "initialized") and self.initialized:
            return
        self.config = graph_config
        self.schema_class: Optional[Type[BaseModel]] = None
        self.fields: list[dict[str, Any]] = []

        # Access schema config from graphConfig object
        schema_config = self.config.schema_config
        if schema_config:
            # Case 1: If 'schema' key is present and non-empty, use the schema class
            # Case 2: If 'schema' key is empty or not present, check if 'fields' are provided
            if "schema" in schema_config and schema_config["schema"]:
                schema_path = schema_config["schema"]
                try:
                    self.schema_class = resolve_schema_class(
                        schema_path
                    )  # Ensure that validators of custom defined class also get executed
                    logger.info(f"Using schema class: {self.schema_class}")
                except ValueError:
                    raise ValueError(f"Invalid schema path: {schema_path}")
            elif not self.schema_class and "fields" in schema_config:
                try:
                    # Get the fields from yaml for validation
                    self.fields = process_custom_fields(schema_config)
                    logger.info(f"Using fields fetched from yaml file: {self.fields}")
                except ValueError:
                    raise ValueError(f"Invalid fields: {self.fields}")
            else:
                raise RuntimeError("Empty schema config")
        else:
            logger.warn("Schema config not defined. Running without validation.")

    def validate_type(self, field_name: str, field_value, expected_type) -> bool:
        """
        Validate that the field's type matches the expected type using Pydantic.
        """
        try:
            # Dynamically create a Pydantic model with the field name and expected type
            # Use TypeAdapter (Pydantic v2) to validate a single value against the expected type
            adapter = TypeAdapter(expected_type)
            adapter.validate_python(field_value)

            # If validation succeeds, it means the field_value is valid
            return True
        except ValidationError as e:
            logger.error(
                f"Validation failed for field '{field_name}': Expected type {expected_type}, but got {type(field_value)}. "
                f"Validation error: {e}"
            )
            return False
        except Exception as e:
            logger.error(f"Error during type validation for field '{field_name}': {e}")
            return False

    def validateYAML(self, data: dict) -> bool:
        """
        Validate data against the schema.
        """
        for field in self.fields:
            try:
                field_name = field["name"]
                field_type = field["type"]
            except KeyError as e:
                logger.error(f"KeyError: Missing key {e} in field dictionary.")
                return False
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return False

            # Check if the field exists in the data
            if field_name not in data:
                logger.error(f"Field '{field_name}' is missing from the input data.")
                return False

            field_value = data.get(field_name)

            # Validate the type
            if not self.validate_type(field_name, field_value, field_type):
                logger.error(
                    f"Field '{field_name}' failed type validation. Expected type: {field_type}"
                )
                return False

            # Check additional validation rules dynamically (if any are present)
            if not self.validate_additional_rules(field_name, field, field_value):
                logger.error(f"Field '{field_name}' failed additional validation rules.")
                return False

        return True  # All checks passed

    def validate_additional_rules(self, field_name: str, field: dict, field_value) -> bool:
        """
        Validate additional rules like `is_greater_than`, `is_not_empty`, etc.
        This method can be extended in subclasses to handle more rules.
        """
        """Dynamically validate additional rules from custom_validations module."""
        try:
            # Look for custom rules in the schema and apply them to the field
            for key, value in field.items():
                if key != "name" and key != "type":
                    rule = key
                    rule_value = value
                    # Check if the method corresponding to the rule exists in custom_validations.py
                    validate_function_name = f"validate_{rule}"

                    validate_function = getattr(custom_validations, validate_function_name, None)
                    if callable(validate_function):
                        # Call the validation function and pass the field value and rule value
                        is_valid = validate_function(field_value, field_name, rule_value)
                        if not is_valid:
                            return False
                    else:
                        logger.error(
                            f"Validation function '{validate_function_name}' not found in custom_validations for field '{field_name}'."
                        )
                        return False
            return True
        except Exception as e:
            logger.error(f"Error during additional validation for field '{field_name}': {e}")
            return False

    def validate(self, data: dict) -> bool:
        """
        Validate the data against the selected schema.
        """
        # Override to disable validation check
        """
           Validates the provided data based on the schema class or custom fields.
           Returns True if valid or if there is no schema, False otherwise.
        """
        if not (self.schema_class or self.fields):
            logger.info("Skipping validation, schema_config not defined.")
            return True
        if not self.schema_class:
            logger.info("YAML file based validation triggered. Routing to validateYAML.")
            return self.validateYAML(data)
        try:
            self.schema_class(**data)  # Validate data using the selected schema
            return True
        except ValidationError as e:
            print("Validation Error:", e)  # Raise error if output data didn't match chosen schema
            return False
