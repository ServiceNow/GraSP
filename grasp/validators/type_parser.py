import types
from typing import Any, Union


class TypeParseError(Exception):
    """Custom exception for type parsing errors"""

    pass


ParsedType = Union[type, types.GenericAlias, Any]


class TypeParser:
    # Map of string names to actual type objects
    VALID_TYPES = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "none": type(None),
        "nonetype": type(None),
        "any": Any,
    }

    def __init__(self):
        self.pos = 0
        self.input = ""

    def parse(self, type_str: str) -> ParsedType:
        """Main parsing function that processes the type string and returns actual type"""
        try:
            # Reset state
            self.pos = 0
            # Normalize input by removing spaces and converting to lowercase
            self.input = type_str.replace(" ", "").lower()

            if not self.input:
                raise TypeParseError("Empty type string")

            result = self._parse_type()

            # Check if we've consumed all input
            if self.pos < len(self.input):
                raise TypeParseError(f"Unexpected characters after position {self.pos}")

            return result

        except IndexError:
            raise TypeParseError("Unexpected end of input - missing closing bracket?")

    def _parse_type(self) -> ParsedType:
        """Parse a single type, which may be nested"""
        # Get the base type
        base_type = ""
        while self.pos < len(self.input) and self.input[self.pos].isalnum():
            base_type += self.input[self.pos]
            self.pos += 1

        if not base_type:
            raise TypeParseError(f"Expected type at position {self.pos}")

        if base_type not in self.VALID_TYPES:
            raise TypeParseError(f"Invalid type '{base_type}'")

        type_obj: ParsedType = self.VALID_TYPES[base_type]

        # Handle nested types
        if self.pos < len(self.input) and self.input[self.pos] == "[":
            self.pos += 1  # Skip '['

            if base_type not in ("list", "dict", "set", "tuple"):
                raise TypeParseError(f"Type '{base_type}' cannot have nested types")

            if base_type == "dict":
                # Parse key type
                key_type = self._parse_type()

                # Expect comma
                if self.pos >= len(self.input) or self.input[self.pos] != ",":
                    raise TypeParseError("Expected ',' after dict key type")
                self.pos += 1

                # Parse value type
                value_type = self._parse_type()

                # Build a parameterized dict type at runtime without confusing static type checkers
                type_obj = dict.__class_getitem__((key_type, value_type))
            else:
                # Parse inner type for list, set, tuple
                inner_type = self._parse_type()
                if base_type == "list":
                    type_obj = list.__class_getitem__(inner_type)
                elif base_type == "set":
                    # Use built-in set's __class_getitem__ to avoid mypy complaints
                    type_obj = set.__class_getitem__(inner_type)
                elif base_type == "tuple":
                    type_obj = tuple.__class_getitem__((inner_type, ...))

            # Expect closing bracket
            if self.pos >= len(self.input) or self.input[self.pos] != "]":
                raise TypeParseError("Expected closing ']'")
            self.pos += 1

        return type_obj
