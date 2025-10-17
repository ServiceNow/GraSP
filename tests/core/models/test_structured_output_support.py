# !/usr/bin/env python3
"""
Simplified Unit Tests for Structured Output Functionality

Key test scenarios:
1. StructuredOutputConfig and SchemaConfigParser
2. Native and fallback structured output methods
3. Error handling with mocked clients
"""

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from langchain_core.messages import AIMessage
from langchain_core.prompt_values import ChatPromptValue
from pydantic import BaseModel, Field, ValidationError

from sygra.core.models.custom_models import (
    BaseCustomModel,
    CustomOllama,
    CustomOpenAI,
    CustomTGI,
    CustomVLLM,
    ModelParams,
)
from sygra.core.models.structured_output.structured_output_config import (
    SchemaConfigParser,
    StructuredOutputConfig,
)


# Test schema
class UserSchema(BaseModel):
    name: str = Field(description="User's name")
    age: int = Field(description="User's age")
    email: str = Field(description="User's email")


# Test model that implements abstract method
class CustomModel(BaseCustomModel):
    @pytest.mark.asyncio
    async def _generate_response(self, input, model_params):
        return "test response", 200

    def _supports_native_structured_output(self):
        # Override to return True for testing
        return True


# Mock client for API responses
class MockClient:
    def __init__(self, response_text="", status_code=200):
        self.response_text = response_text
        self.status_code = status_code

    def build_request(self, **kwargs):
        return kwargs

    def build_request_with_payload(self, payload: Dict[str, Any], **kwargs):
        return payload

    async def async_send_request(self, payload, **kwargs):
        response = Mock()
        response.text = self.response_text
        response.status_code = self.status_code
        return response

    async def send_request(self, payload, model_name=None, extra_params=None, **kwargs):
        # Create a simple response object instead of using Mock
        class MockResponse:
            def __init__(self, text, status_code):
                self.text = text
                self.status_code = status_code
                self.choices = [Mock()]
                self.choices[0].model_dump = lambda: {"message": {"content": text}}

            def __getattr__(self, name):
                if name == "status_code":
                    return self.status_code
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        return MockResponse(self.response_text, self.status_code)


class TestSchemaConfigParser(unittest.TestCase):
    """Test SchemaConfigParser initialization and parsing"""

    def test_parse_class_path_valid(self):
        """Test parsing valid class path"""
        config = {"schema": "tests.core.models.test_structured_output_support.UserSchema"}

        with patch.object(SchemaConfigParser, "_import_class", return_value=UserSchema):
            parser = SchemaConfigParser(config)

            self.assertEqual(parser.schema_type, "class")
            self.assertEqual(
                parser.class_path,
                "tests.core.models.test_structured_output_support.UserSchema",
            )
            self.assertIsNone(parser.schema_data)

    def test_parse_schema_dict_valid(self):
        """Test parsing valid schema dictionary"""
        config = {
            "schema": {
                "fields": {
                    "name": {"type": "str", "description": "User name"},
                    "age": {"type": "int", "description": "User age"},
                }
            }
        }

        parser = SchemaConfigParser(config)

        self.assertEqual(parser.schema_type, "schema")
        self.assertEqual(parser.schema_data["fields"]["name"]["type"], "str")
        self.assertIsNone(parser.class_path)

    def test_missing_schema_raises_error(self):
        """Test missing schema field raises ValueError"""
        with self.assertRaises(ValueError) as context:
            SchemaConfigParser({})
        self.assertIn("Schema field is required", str(context.exception))

    def test_invalid_schema_type_raises_error(self):
        """Test invalid schema type raises ValueError"""
        with self.assertRaises(ValueError) as context:
            SchemaConfigParser({"schema": 123})
        self.assertIn("Schema must be either a string", str(context.exception))

    def test_invalid_class_path_format(self):
        """Test invalid class path format"""
        config = {"schema": "invalid_format"}

        with self.assertRaises(ValueError) as context:
            SchemaConfigParser(config)
        self.assertIn("Invalid class path format", str(context.exception))

    def test_schema_dict_missing_fields(self):
        """Test schema dict without fields key"""
        config = {"schema": {"name": "TestSchema"}}

        with self.assertRaises(ValueError) as context:
            SchemaConfigParser(config)
        self.assertIn("must contain 'fields' key", str(context.exception))


class TestStructuredOutputConfig(unittest.TestCase):
    """Test StructuredOutputConfig initialization and methods"""

    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    def test_config_enabled_by_default(self, mock_parser):
        """Test config is enabled by default when key present"""
        config = StructuredOutputConfig({})
        self.assertTrue(config.enabled)
        self.assertEqual(config.fallback_strategy, "instruction")
        self.assertEqual(config.max_parse_retries, 2)

    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    def test_config_disabled_explicitly(self, mock_parser):
        """Test config can be disabled explicitly"""
        config = StructuredOutputConfig({"enabled": False})
        self.assertFalse(config.enabled)

    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    def test_custom_fallback_strategy(self, mock_parser):
        """Test custom fallback strategy"""
        config = StructuredOutputConfig({"fallback_strategy": "post_process"})
        self.assertEqual(config.fallback_strategy, "post_process")

    @patch.object(StructuredOutputConfig, "_load_class_from_path")
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    def test_get_pydantic_model_class_path(self, mock_parser, mock_load):
        """Test getting pydantic model from class path"""
        # Setup mock parser
        mock_parser_instance = Mock()
        mock_parser_instance.schema_type = "class"
        mock_parser_instance.class_path = "test.TestSchema"
        mock_parser_instance.schema_data = None
        mock_parser.return_value = mock_parser_instance

        mock_load.return_value = UserSchema

        config = StructuredOutputConfig({"schema": "test.TestSchema"})
        result = config.get_pydantic_model()

        self.assertEqual(result, UserSchema)
        mock_load.assert_called_once_with("test.TestSchema")

    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    def test_get_pydantic_model_disabled(self, mock_parser):
        """Test get_pydantic_model returns None when disabled"""
        config = StructuredOutputConfig({"enabled": False})
        result = config.get_pydantic_model()
        self.assertIsNone(result)

    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    def test_python_type_mapping(self, mock_parser):
        """Test type string to Python type conversion"""
        config = StructuredOutputConfig({})

        test_cases = [
            ("str", str),
            ("string", str),
            ("int", int),
            ("integer", int),
            ("float", float),
            ("bool", bool),
            ("list", list),
            ("dict", dict),
            ("unknown", str),  # default case
        ]

        for type_str, expected_type in test_cases:
            self.assertEqual(config._get_python_type(type_str), expected_type)


class TestStructuredOutputMethods(unittest.IsolatedAsyncioTestCase):
    """Test structured output methods in custom models"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            "name": "test-model",
            "parameters": {"temperature": 0.7},
            "structured_output": {"enabled": True, "schema": "test.TestSchema"},
        }
        self.test_input = ChatPromptValue(messages=[AIMessage(content="Generate user info")])
        self.test_params = ModelParams(url="https://test-url.com", auth_token="test-token")
        self.valid_json = '{"name": "Test User", "age": 30, "email": "test@example.com"}'

    @patch("sygra.utils.utils.validate_required_keys")
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_supports_native_structured_output(self, mock_parser, mock_utils):
        """Test native structured output support detection"""
        openai_model = CustomOpenAI(
            {
                **self.test_config,
                "url": "test",
                "api_key": "test",
                "api_version": "test",
                "model": "gpt-4",
            }
        )
        vllm_model = CustomVLLM({**self.test_config, "url": "test", "auth_token": "test"})
        tgi_model = CustomTGI({**self.test_config, "url": "test", "auth_token": "test"})
        base_model = CustomModel(self.test_config)

        self.assertTrue(openai_model._supports_native_structured_output())
        self.assertTrue(vllm_model._supports_native_structured_output())
        self.assertTrue(tgi_model._supports_native_structured_output())
        self.assertTrue(
            base_model._supports_native_structured_output()
        )  # Base model returns True for these types

    @patch("sygra.utils.utils.validate_required_keys")
    @patch("sygra.core.models.custom_models.PydanticOutputParser")
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_fallback_structured_output(self, mock_parser, mock_output_parser, mock_utils):
        """Test fallback structured output generation"""
        # Setup mocks
        mock_parser_instance = Mock()
        mock_parser_instance.get_format_instructions.return_value = "Format as JSON"
        mock_parser_instance.parse.return_value = UserSchema(
            name="Test", age=30, email="test@example.com"
        )
        mock_output_parser.return_value = mock_parser_instance

        model = CustomModel(self.test_config)
        model._generate_response_with_retry = AsyncMock(return_value=(self.valid_json, 200))

        # Execute
        resp_text, resp_status = await model._generate_fallback_structured_output(
            self.test_input, self.test_params, UserSchema
        )

        # Verify
        self.assertEqual(resp_status, 200)
        parsed_data = json.loads(resp_text)
        self.assertEqual(parsed_data["name"], "Test")

    @patch("sygra.utils.utils.validate_required_keys")
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_openai_native_structured_output_success(self, mock_parser, mock_utils):
        """Test OpenAI native structured output success"""
        model = CustomOpenAI(
            {
                **self.test_config,
                "url": "test",
                "api_key": "test",
                "api_version": "test",
                "model": "gpt-4",
            }
        )
        model._client = MockClient(response_text=self.valid_json, status_code=200)

        # Mock _set_client to prevent it from overwriting our mock client
        with (
            patch.object(model, "_set_client"),
            patch(
                "pydantic.BaseModel.model_validate",
                return_value=UserSchema(name="Test", age=30, email="test@example.com"),
            ),
        ):
            resp_text, resp_status = await model._generate_native_structured_output(
                self.test_input, self.test_params, UserSchema
            )

            self.assertEqual(resp_status, 200)
            self.assertIn("Test", resp_text)

    @patch("sygra.utils.utils.validate_required_keys")
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_openai_native_structured_output_http_error(self, mock_parser, mock_utils):
        """Test OpenAI native structured output with HTTP error"""
        model = CustomOpenAI(
            {
                **self.test_config,
                "url": "test",
                "api_key": "test",
                "api_version": "test",
                "model": "gpt-4",
            }
        )
        model._client = MockClient(response_text="Error", status_code=500)
        model._generate_fallback_structured_output = AsyncMock(return_value=(self.valid_json, 200))

        # Mock _set_client to prevent it from overwriting our mock client
        with patch.object(model, "_set_client"):
            resp_text, resp_status = await model._generate_native_structured_output(
                self.test_input, self.test_params, UserSchema
            )

            # Should fallback
            model._generate_fallback_structured_output.assert_called_once()
            self.assertEqual(resp_status, 200)

    @patch("sygra.utils.utils.validate_required_keys")
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_vllm_native_structured_output_success(self, mock_parser, mock_utils):
        """Test VLLM native structured output success"""
        model = CustomVLLM({**self.test_config, "url": "test", "auth_token": "test"})
        model._client = MockClient(response_text=self.valid_json, status_code=200)

        # Mock _set_client to prevent it from overwriting our mock client
        with (
            patch.object(model, "_set_client"),
            patch(
                "pydantic.BaseModel.model_validate",
                return_value=UserSchema(name="Test", age=30, email="test@example.com"),
            ),
        ):
            resp_text, resp_status = await model._generate_native_structured_output(
                self.test_input, self.test_params, UserSchema
            )

            self.assertEqual(resp_status, 200)

    @patch("sygra.utils.utils.validate_required_keys")
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_vllm_native_structured_output_validation_error(self, mock_parser, mock_utils):
        """Test VLLM native structured output with validation error"""
        model = CustomVLLM({**self.test_config, "url": "test", "auth_token": "test"})
        model._client = MockClient(
            response_text='{"name": "Test", "age": "invalid"}', status_code=200
        )
        model._generate_fallback_structured_output = AsyncMock(return_value=(self.valid_json, 200))

        # Mock validation to raise error
        def mock_validate(*args, **kwargs):
            raise ValidationError.from_exception_data(
                "UserSchema",
                [
                    {
                        "type": "int_parsing",
                        "loc": ("age",),
                        "msg": "Input should be a valid integer",
                        "input": "invalid",
                    }
                ],
            )

        # Mock _set_client to prevent it from overwriting our mock client
        with (
            patch.object(model, "_set_client"),
            patch("pydantic.BaseModel.model_validate", side_effect=mock_validate),
        ):
            resp_text, resp_status = await model._generate_native_structured_output(
                self.test_input, self.test_params, UserSchema
            )

        # Should fallback
        model._generate_fallback_structured_output.assert_called_once()

    @patch("sygra.utils.utils.validate_required_keys")
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_tgi_native_structured_output_success(self, mock_parser, mock_utils):
        """Test TGI native structured output success"""
        model = CustomTGI({**self.test_config, "url": "test", "auth_token": "test"})
        tgi_response = json.dumps({"generated_text": self.valid_json})
        model._client = MockClient(response_text=tgi_response, status_code=200)

        # Mock missing tokenizer attribute
        model.tokenizer = Mock()

        # Mock _set_client to prevent it from overwriting our mock client
        with (
            patch.object(model, "_set_client"),
            patch(
                "pydantic.BaseModel.model_validate",
                return_value=UserSchema(name="Test", age=30, email="test@example.com"),
            ),
        ):
            resp_text, resp_status = await model._generate_native_structured_output(
                self.test_input, self.test_params, UserSchema
            )

            self.assertEqual(resp_status, 200)
            # TGI returns parsed dictionary, not JSON string
            self.assertEqual(resp_text, self.valid_json)

    @patch("sygra.utils.utils.validate_required_keys")
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_tgi_native_structured_output_http_error(self, mock_parser, mock_utils):
        """Test TGI native structured output with HTTP error"""
        model = CustomTGI({**self.test_config, "url": "test", "auth_token": "test"})
        model._client = MockClient(response_text="Server Error", status_code=500)
        model._generate_fallback_structured_output = AsyncMock(return_value=(self.valid_json, 200))

        # Mock missing tokenizer attribute
        model.tokenizer = Mock()

        # Mock _set_client to prevent it from overwriting our mock client
        with patch.object(model, "_set_client"):
            resp_text, resp_status = await model._generate_native_structured_output(
                self.test_input, self.test_params, UserSchema
            )

            # Should fallback
            model._generate_fallback_structured_output.assert_called_once()
            self.assertEqual(resp_status, 200)

    @patch("sygra.utils.utils.validate_required_keys")
    @patch(
        "sygra.core.models.structured_output.structured_output_config.StructuredOutputConfig.get_pydantic_model"
    )
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_handle_structured_output_with_native_support(
        self, mock_parser, mock_get_model, mock_utils
    ):
        """Test _handle_structured_output with native support"""
        mock_get_model.return_value = UserSchema

        model = CustomOpenAI(
            {
                **self.test_config,
                "url": "test",
                "api_key": "test",
                "api_version": "test",
                "model": "gpt-4",
            }
        )
        model._generate_native_structured_output = AsyncMock(return_value=(self.valid_json, 200))

        # Create a simple mock lock
        async def mock_lock():
            pass

        model._structured_output_lock = Mock()
        model._structured_output_lock.__aenter__ = AsyncMock(return_value=None)
        model._structured_output_lock.__aexit__ = AsyncMock(return_value=None)

        resp_text, resp_status = await model._handle_structured_output(
            self.test_input, self.test_params
        )

        self.assertEqual(resp_text, self.valid_json)
        self.assertEqual(resp_status, 200)

    @patch("sygra.utils.utils.validate_required_keys")
    @patch(
        "sygra.core.models.structured_output.structured_output_config.StructuredOutputConfig.get_pydantic_model"
    )
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_handle_structured_output_disabled(self, mock_parser, mock_get_model, mock_utils):
        """Test _handle_structured_output when disabled"""
        mock_get_model.return_value = None  # No valid schema

        model = CustomModel(self.test_config)

        # Create a simple mock lock
        model._structured_output_lock = Mock()
        model._structured_output_lock.__aenter__ = AsyncMock(return_value=None)
        model._structured_output_lock.__aexit__ = AsyncMock(return_value=None)

        resp = await model._handle_structured_output(self.test_input, self.test_params)

        self.assertIsNone(resp)

    @patch("sygra.utils.utils.validate_required_keys")
    @patch(
        "sygra.core.models.structured_output.structured_output_config.StructuredOutputConfig.get_pydantic_model"
    )
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_handle_structured_output_fallback_mode(
        self, mock_parser, mock_get_model, mock_utils
    ):
        """Test _handle_structured_output using fallback for unsupported model"""
        mock_get_model.return_value = UserSchema

        # Create a model that doesn't support native structured output
        model = CustomModel(self.test_config)
        model._supports_native_structured_output = lambda: False
        model._generate_fallback_structured_output = AsyncMock(return_value=(self.valid_json, 200))

        # Create a simple mock lock
        model._structured_output_lock = Mock()
        model._structured_output_lock.__aenter__ = AsyncMock(return_value=None)
        model._structured_output_lock.__aexit__ = AsyncMock(return_value=None)

        resp_text, resp_status = await model._handle_structured_output(
            self.test_input, self.test_params
        )

        model._generate_fallback_structured_output.assert_called_once()
        self.assertEqual(resp_text, self.valid_json)
        self.assertEqual(resp_status, 200)

    @patch("sygra.utils.utils.validate_required_keys")
    @patch("sygra.core.models.custom_models.PydanticOutputParser")
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_fallback_structured_output_parse_error(
        self, mock_parser, mock_output_parser, mock_utils
    ):
        """Test fallback structured output with parsing error"""
        # Setup mocks
        mock_parser_instance = Mock()
        mock_parser_instance.get_format_instructions.return_value = "Format as JSON"
        mock_parser_instance.parse.side_effect = Exception("Parse error")
        mock_output_parser.return_value = mock_parser_instance

        model = CustomModel(self.test_config)
        model._generate_response_with_retry = AsyncMock(return_value=("Invalid JSON", 200))

        # Execute
        resp_text, resp_status = await model._generate_fallback_structured_output(
            self.test_input, self.test_params, UserSchema
        )

        # Should return unparsed response when parsing fails
        self.assertEqual(resp_text, "Invalid JSON")
        self.assertEqual(resp_status, 200)

    @patch("sygra.utils.utils.validate_required_keys")
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_ollama_completions_api_response_extraction(self, mock_parser, mock_utils):
        """Test Ollama response extraction for completions API"""
        # Test completions API response handling
        completions_config = {
            **self.test_config,
            "url": "test",
            "auth_token": "test",
            "completions_api": True,
        }
        model = CustomOllama(completions_config)

        # Mock missing tokenizer attribute
        model.tokenizer = Mock()

        # Mock response for completions API
        mock_response = {"response": self.valid_json}
        model._client = Mock()
        model._client.send_request = AsyncMock(return_value=mock_response)

        # Mock _set_client to prevent it from overwriting our mock client
        with (
            patch.object(model, "_set_client"),
            patch(
                "pydantic.BaseModel.model_validate",
                return_value=UserSchema(name="Test", age=30, email="test@example.com"),
            ),
        ):
            resp_text, resp_status = await model._generate_native_structured_output(
                self.test_input, self.test_params, UserSchema
            )

            # Verify the response tuple
            self.assertEqual(resp_text, self.valid_json)
            self.assertEqual(resp_status, 200)

    @patch("sygra.utils.utils.validate_required_keys")
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_ollama_chat_api_response_extraction(self, mock_parser, mock_utils):
        """Test Ollama response extraction for chat API"""
        # Test chat API response handling (default)
        model = CustomOllama({**self.test_config, "url": "test", "auth_token": "test"})

        # Mock missing tokenizer attribute
        model.tokenizer = Mock()

        # Mock response for chat API
        mock_response = {"message": {"content": self.valid_json}}
        model._client = Mock()
        model._client.send_request = AsyncMock(return_value=mock_response)

        # Mock _set_client to prevent it from overwriting our mock client
        with (
            patch.object(model, "_set_client"),
            patch(
                "pydantic.BaseModel.model_validate",
                return_value=UserSchema(name="Test", age=30, email="test@example.com"),
            ),
        ):
            resp_text, resp_status = await model._generate_native_structured_output(
                self.test_input, self.test_params, UserSchema
            )

            # Verify the response tuple
            self.assertEqual(resp_text, self.valid_json)
            self.assertEqual(resp_status, 200)

    @patch("sygra.utils.utils.validate_required_keys")
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_ollama_generate_response_success(self, mock_parser, mock_utils):
        """Test Ollama regular text generation success"""
        model = CustomOllama({**self.test_config, "url": "test", "auth_token": "test"})

        # Mock response for chat API
        mock_response = {"message": {"content": "Generated text response"}}
        model._client = Mock()
        model._client.send_request = AsyncMock(return_value=mock_response)

        # Mock _set_client to prevent it from overwriting our mock client
        with patch.object(model, "_set_client"):
            resp_text, resp_status = await model._generate_response(self.test_input, self.test_params)

            self.assertEqual(resp_text, "Generated text response")
            self.assertEqual(resp_status, 200)

    @patch("sygra.utils.utils.validate_required_keys")
    @patch("sygra.core.models.structured_output.structured_output_config.SchemaConfigParser")
    async def test_ollama_generate_response_exception_handling(self, mock_parser, mock_utils):
        """Test Ollama regular text generation exception handling"""
        model = CustomOllama({**self.test_config, "url": "test", "auth_token": "test"})

        # Mock _set_client to raise an exception
        with patch.object(model, "_set_client", side_effect=Exception("Connection failed")):
            resp_text, resp_status = await model._generate_response(self.test_input, self.test_params)

            self.assertIn("ERROR", resp_text)
            self.assertIn("Connection failed", resp_text)
            self.assertEqual(resp_status, 999)


if __name__ == "__main__":
    unittest.main()
