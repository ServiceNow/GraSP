import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the parent directory to sys.path to import the necessary modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from sygra.core.models.custom_models import CustomTriton, ModelParams
from sygra.utils import constants


class TestCustomTriton(unittest.TestCase):
    """Unit tests for the CustomTriton class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Base model configuration
        self.base_config = {
            "name": "triton_model",
            "parameters": {"temperature": 0.7, "max_tokens": 100},
            "url": "http://triton-test.com",
            "auth_token": "Bearer test_token",
            "payload_key": "default",
        }

        # Mock messages
        self.messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello, how are you?"),
        ]
        self.chat_input = ChatPromptValue(messages=self.messages)

        # Mock payloads
        self.mock_payload_config = {
            "payload_json": {
                "inputs": [
                    {"name": "request", "data": []},
                    {"name": "options", "data": []},
                ]
            },
            "response_key": "response",
        }

        self.mock_payload_json = {
            "inputs": [{"name": "request", "data": []}, {"name": "options", "data": []}]
        }

    def test_init(self):
        """Test initialization of CustomTriton"""
        # Create the custom model
        custom_triton = CustomTriton(self.base_config)

        # Verify model was properly initialized
        self.assertEqual(custom_triton.model_config, self.base_config)
        self.assertEqual(custom_triton.generation_params, self.base_config["parameters"])
        self.assertEqual(custom_triton.auth_token, "test_token")
        self.assertEqual(custom_triton.name(), "triton_model")

    @patch("sygra.core.models.custom_models.utils.get_payload")
    def test_get_payload_config_template_default(self, mock_get_payload):
        """Test _get_payload_config_template method with default payload key"""
        mock_get_payload.return_value = self.mock_payload_config

        custom_triton = CustomTriton(self.base_config)
        result = custom_triton._get_payload_config_template(
            constants.INFERENCE_SERVER_TRITON, "default"
        )

        self.assertEqual(result, self.mock_payload_config)
        mock_get_payload.assert_called_once_with(constants.INFERENCE_SERVER_TRITON, "default")

    @patch("sygra.core.models.custom_models.utils.get_payload")
    def test_get_payload_config_template_no_key(self, mock_get_payload):
        """Test _get_payload_config_template method with no payload key"""
        mock_get_payload.return_value = self.mock_payload_config

        custom_triton = CustomTriton(self.base_config)
        result = custom_triton._get_payload_config_template(constants.INFERENCE_SERVER_TRITON, None)

        self.assertEqual(result, self.mock_payload_config)
        mock_get_payload.assert_called_once_with(constants.INFERENCE_SERVER_TRITON)

    @patch("sygra.core.models.custom_models.utils.get_payload")
    def test_get_payload_config_template_exception(self, mock_get_payload):
        """Test _get_payload_config_template method with exception"""
        mock_get_payload.side_effect = Exception("Test error")

        custom_triton = CustomTriton(self.base_config)

        with self.assertRaises(Exception) as context:
            custom_triton._get_payload_config_template(constants.INFERENCE_SERVER_TRITON, "default")

        self.assertTrue("Failed to get payload config: Test error" in str(context.exception))

    def test_get_payload_json_template(self):
        """Test _get_payload_json_template method"""
        custom_triton = CustomTriton(self.base_config)
        result = custom_triton._get_payload_json_template(self.mock_payload_config)

        self.assertEqual(result, self.mock_payload_config["payload_json"])

    def test_get_payload_json_template_missing(self):
        """Test _get_payload_json_template method with missing payload JSON"""
        custom_triton = CustomTriton(self.base_config)

        with self.assertRaises(AssertionError) as context:
            custom_triton._get_payload_json_template({})

        self.assertTrue("Payload JSON must be defined for Triton server" in str(context.exception))

    def test_create_triton_request(self):
        """Test _create_triton_request method"""
        custom_triton = CustomTriton(self.base_config)

        messages = [{"role": "user", "content": "Hello"}]
        generation_params = {"temperature": 0.7}

        result = custom_triton._create_triton_request(
            self.mock_payload_json.copy(), messages, generation_params
        )

        # Verify the request payload was constructed correctly
        self.assertEqual(len(result["inputs"]), 2)
        self.assertEqual(result["inputs"][0]["name"], "request")
        self.assertEqual(result["inputs"][0]["data"], [json.dumps(messages, ensure_ascii=False)])
        self.assertEqual(result["inputs"][1]["name"], "options")
        self.assertEqual(
            result["inputs"][1]["data"],
            [json.dumps(generation_params, ensure_ascii=False)],
        )

    def test_get_response_text_success(self):
        """Test _get_response_text method with successful response"""
        custom_triton = CustomTriton(self.base_config)

        mock_response = json.dumps(
            {"outputs": [{"data": [json.dumps({"response": "Hello there!"})]}]}
        )

        result = custom_triton._get_response_text(mock_response, self.mock_payload_config)

        self.assertEqual(result, "Hello there!")

    def test_get_response_text_dict_response(self):
        """Test _get_response_text method with dict response"""
        custom_triton = CustomTriton(self.base_config)

        mock_response = json.dumps({"outputs": [{"data": [{"response": "Hello there!"}]}]})

        result = custom_triton._get_response_text(mock_response, self.mock_payload_config)

        self.assertEqual(result, "Hello there!")

    @patch("sygra.core.models.custom_models.logger")
    def test_get_response_text_json_error(self, mock_logger):
        """Test _get_response_text method with JSON error"""
        custom_triton = CustomTriton(self.base_config)

        # Mock an error response with invalid JSON returned by model
        mock_response = json.dumps(
            {
                "error": json.dumps(
                    {
                        "error": json.dumps(
                            {
                                "error_message": "Invalid JSON returned by model.",
                                "model_output": "Raw model output here",
                            }
                        )
                    }
                )
            }
        )

        result = custom_triton._get_response_text(mock_response, self.mock_payload_config)

        self.assertEqual(result, "Raw model output here")
        mock_logger.error.assert_any_call("Invalid JSON returned, JSON mode specified")

    @patch("sygra.core.models.custom_models.logger")
    def test_get_response_text_non_json_error(self, mock_logger):
        """Test _get_response_text method with non-JSON error"""
        custom_triton = CustomTriton(self.base_config)

        # Mock an error response with non-JSON error
        mock_response = json.dumps(
            {
                "error": json.dumps(
                    {
                        "error": json.dumps(
                            {
                                "error_message": "Some other error",
                            }
                        )
                    }
                )
            }
        )

        result = custom_triton._get_response_text(mock_response, self.mock_payload_config)

        self.assertEqual(result, "")
        mock_logger.error.assert_any_call("Not a JSON error. Error message: Some other error")

    @patch("sygra.core.models.custom_models.logger")
    def test_get_response_text_parse_exception(self, mock_logger):
        """Test _get_response_text method with parse exception"""
        custom_triton = CustomTriton(self.base_config)

        # Invalid JSON
        mock_response = "This is not JSON"

        result = custom_triton._get_response_text(mock_response, self.mock_payload_config)

        self.assertEqual(result, "")
        mock_logger.error.assert_any_call(
            "Failed to get response text: Expecting value: line 1 column 1 (char 0)"
        )

    @patch("sygra.core.models.custom_models.utils")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_success(self, mock_set_client, mock_utils):
        """Test _generate_response method with successful response"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request.return_value = {"payload": "test_payload"}
        mock_client.async_send_request = AsyncMock()

        # Configure the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(
            {"outputs": [{"data": [json.dumps({"response": "Hello there!"})]}]}
        )
        mock_client.async_send_request.return_value = mock_response

        # Mock utils methods
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = [
            {"role": "user", "content": "Hello"}
        ]
        mock_utils.get_payload.return_value = self.mock_payload_config

        # Setup custom model
        custom_triton = CustomTriton(self.base_config)
        custom_triton._client = mock_client

        # Patch internal methods to avoid actual calls
        custom_triton._get_payload_config_template = MagicMock(
            return_value=self.mock_payload_config
        )
        custom_triton._get_payload_json_template = MagicMock(return_value=self.mock_payload_json)
        custom_triton._create_triton_request = MagicMock(return_value=self.mock_payload_json)

        # Call _generate_response
        model_params = ModelParams(url="http://triton-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_triton._generate_response(self.chat_input, model_params)

        # Verify results
        self.assertEqual(resp_text, "Hello there!")
        self.assertEqual(resp_status, 200)

        # Verify method calls
        mock_set_client.assert_called_once()
        custom_triton._get_payload_config_template.assert_called_once_with(
            constants.INFERENCE_SERVER_TRITON, "default"
        )
        custom_triton._get_payload_json_template.assert_called_once_with(self.mock_payload_config)
        mock_client.build_request.assert_called_once()
        mock_client.async_send_request.assert_awaited_once()

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.utils")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_http_error(self, mock_set_client, mock_utils, mock_logger):
        """Test _generate_response method with HTTP error"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request.return_value = {"payload": "test_payload"}
        mock_client.async_send_request = AsyncMock()

        # Configure the mock response with error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client.async_send_request.return_value = mock_response

        # Mock utils methods
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = [
            {"role": "user", "content": "Hello"}
        ]
        mock_utils.get_payload.return_value = self.mock_payload_config

        # Setup custom model
        custom_triton = CustomTriton(self.base_config)
        custom_triton._client = mock_client
        custom_triton._get_status_from_body = MagicMock(return_value=500)

        # Patch internal methods to avoid actual calls
        custom_triton._get_payload_config_template = MagicMock(
            return_value=self.mock_payload_config
        )
        custom_triton._get_payload_json_template = MagicMock(return_value=self.mock_payload_json)
        custom_triton._create_triton_request = MagicMock(return_value=self.mock_payload_json)

        # Call _generate_response
        model_params = ModelParams(url="http://triton-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_triton._generate_response(self.chat_input, model_params)

        # Verify results
        self.assertEqual(resp_text, "")
        self.assertEqual(resp_status, 500)

        # Verify error logging
        mock_logger.error.assert_called_with(
            "[triton_model] HTTP request failed with code: 500 and error: Internal Server Error"
        )

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_exception(self, mock_set_client, mock_logger):
        """Test _generate_response method with exception"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request.side_effect = Exception("Test error")

        # Setup custom model
        custom_triton = CustomTriton(self.base_config)
        custom_triton._client = mock_client
        custom_triton._get_status_from_body = MagicMock(return_value=None)

        # Call _generate_response
        model_params = ModelParams(url="http://triton-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_triton._generate_response(self.chat_input, model_params)

        # Verify results
        self.assertEqual(resp_text, "Http request failed Test error")
        self.assertEqual(resp_status, 999)

        # Verify error logging
        mock_logger.error.assert_called_with("Http request failed Test error")


if __name__ == "__main__":
    unittest.main()
