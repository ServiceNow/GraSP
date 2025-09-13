import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path to import the necessary modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from pydantic import BaseModel

from grasp.core.models.client.openai_azure_client import (
    AzureClientConfig,
    OpenAIAzureClient,
)


class SampleStructuredOutput(BaseModel):
    name: str
    age: int
    skills: list[str]


class TestOpenaiAzureClient(unittest.TestCase):
    """Unit tests for the OpenaiAzureClient class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.valid_config = {
            "azure_deployment": "gpt-4",
            "azure_endpoint": "https://test-endpoint.openai.azure.com",
            "api_version": "2023-05-15",
            "api_key": "test-api-key",
            "timeout": 90,
            "max_retries": 2,
        }

    @patch("grasp.core.models.client.openai_azure_client.AsyncAzureOpenAI")
    def test_init_async_client(self, mock_async_openai):
        """Test initialization of OpenaiAzureClient with async client"""
        mock_async_openai.return_value = MagicMock()

        client = OpenAIAzureClient(async_client=True, **self.valid_config)

        # Verify client was properly initialized
        self.assertTrue(client.async_client)
        self.assertTrue(client.chat_completions_api)
        self.assertIsNone(client.stop)
        mock_async_openai.assert_called_once()

        # Check that the config was validated and passed correctly
        args, kwargs = mock_async_openai.call_args
        self.assertEqual(kwargs["azure_deployment"], "gpt-4")
        self.assertEqual(kwargs["azure_endpoint"], "https://test-endpoint.openai.azure.com")
        self.assertEqual(kwargs["api_version"], "2023-05-15")
        self.assertEqual(kwargs["api_key"], "test-api-key")
        self.assertEqual(kwargs["timeout"], 90)
        self.assertEqual(kwargs["max_retries"], 2)
        self.assertEqual(kwargs["default_headers"], {"Connection": "close"})

    @patch("grasp.core.models.client.openai_azure_client.AzureOpenAI")
    def test_init_sync_client(self, mock_openai):
        """Test initialization of OpenaiAzureClient with sync client"""
        mock_openai.return_value = MagicMock()

        client = OpenAIAzureClient(async_client=False, **self.valid_config)

        # Verify client was properly initialized
        self.assertFalse(client.async_client)
        self.assertTrue(client.chat_completions_api)
        mock_openai.assert_called_once()

        # Check that the config was validated and passed correctly
        args, kwargs = mock_openai.call_args
        self.assertEqual(kwargs["azure_deployment"], "gpt-4")
        self.assertEqual(kwargs["azure_endpoint"], "https://test-endpoint.openai.azure.com")

    @patch("grasp.core.models.client.openai_azure_client.AzureOpenAI")
    def test_init_with_stop_sequence(self, mock_openai):
        """Test initialization with stop sequence"""
        mock_openai.return_value = MagicMock()

        stop_sequence = ["END", "STOP"]
        client = OpenAIAzureClient(async_client=False, stop=stop_sequence, **self.valid_config)

        # Verify stop sequence was set
        self.assertEqual(client.stop, stop_sequence)

    @patch("grasp.core.models.client.openai_azure_client.AzureOpenAI")
    def test_init_with_completions_api(self, mock_openai):
        """Test initialization with completion API flag"""
        mock_openai.return_value = MagicMock()

        client = OpenAIAzureClient(
            async_client=False, chat_completions_api=False, **self.valid_config
        )

        # Verify chat_completions_api flag was set
        self.assertFalse(client.chat_completions_api)

    def test_convert_input_with_prompt_value(self):
        """Test _convert_input with a PromptValue"""
        MagicMock(spec=OpenAIAzureClient)

        # Call the static method directly
        prompt_value = StringPromptValue(text="Test prompt")
        result = OpenAIAzureClient._convert_input(prompt_value)

        # Verify the result is the same PromptValue
        self.assertEqual(result, prompt_value)

    def test_convert_input_with_string(self):
        """Test _convert_input with a string"""
        MagicMock(spec=OpenAIAzureClient)

        # Call the static method directly
        result = OpenAIAzureClient._convert_input("Test prompt")

        # Verify the result is a StringPromptValue
        self.assertIsInstance(result, StringPromptValue)
        self.assertEqual(result.text, "Test prompt")

    def test_convert_input_with_messages(self):
        """Test _convert_input with a list of messages"""
        MagicMock(spec=OpenAIAzureClient)

        # Create a list of messages
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello, how are you?"),
            AIMessage(content="I'm doing well, thank you!"),
        ]

        # Call the static method directly
        result = OpenAIAzureClient._convert_input(messages)

        # Verify the result is a ChatPromptValue
        self.assertIsInstance(result, ChatPromptValue)
        self.assertEqual(len(result.messages), 3)
        self.assertEqual(result.messages[0].content, "You are a helpful assistant")
        self.assertEqual(result.messages[1].content, "Hello, how are you?")
        self.assertEqual(result.messages[2].content, "I'm doing well, thank you!")

    def test_convert_input_with_invalid_type(self):
        """Test _convert_input with an invalid input type"""
        MagicMock(spec=OpenAIAzureClient)

        # Call the static method with an invalid type
        with self.assertRaises(ValueError) as context:
            OpenAIAzureClient._convert_input(123)

        # Verify the error message
        self.assertIn("Invalid input type", str(context.exception))
        self.assertIn(
            "Must be a PromptValue, str, or list of BaseMessages",
            str(context.exception),
        )

    @patch("grasp.core.models.client.openai_azure_client.AzureOpenAI")
    def test_build_request_chat_completions(self, mock_openai):
        """Test build_request with chat completions API"""
        mock_openai.return_value = MagicMock()

        client = OpenAIAzureClient(async_client=False, **self.valid_config)

        # Create a list of messages
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello, how are you?"),
        ]

        # Build the request
        payload = client.build_request(messages=messages, temperature=0.7, max_tokens=100)

        # Verify the payload
        self.assertIn("messages", payload)
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "You are a helpful assistant")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"], "Hello, how are you?")
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_tokens"], 100)

    @patch("grasp.core.models.client.openai_azure_client.AzureOpenAI")
    def test_build_request_chat_completions_with_stop(self, mock_openai):
        """Test build_request with chat completions API and stop sequence"""
        mock_openai.return_value = MagicMock()

        stop_sequence = ["END", "STOP"]
        client = OpenAIAzureClient(async_client=False, stop=stop_sequence, **self.valid_config)

        # Create a list of messages
        messages = [HumanMessage(content="Hello, how are you?")]

        # Build the request
        payload = client.build_request(messages=messages)

        # Verify the payload includes stop sequence
        self.assertIn("stop", payload)
        self.assertEqual(payload["stop"], stop_sequence)

    @patch("grasp.core.models.client.openai_azure_client.AzureOpenAI")
    def test_build_request_chat_completions_invalid_messages(self, mock_openai):
        """Test build_request with chat completions API and invalid messages"""
        mock_openai.return_value = MagicMock()

        client = OpenAIAzureClient(async_client=False, **self.valid_config)

        # Try to build request with empty messages
        with self.assertRaises(ValueError) as context:
            client.build_request(messages=[])

        # Verify the error message
        self.assertIn("messages passed is None or empty", str(context.exception))

        # Try to build request with None messages
        with self.assertRaises(ValueError) as context:
            client.build_request(messages=None)

        # Verify the error message
        self.assertIn("messages passed is None or empty", str(context.exception))

    @patch("grasp.core.models.client.openai_azure_client.AzureOpenAI")
    def test_build_request_completions(self, mock_openai):
        """Test build_request with completions API"""
        mock_openai.return_value = MagicMock()

        client = OpenAIAzureClient(
            async_client=False, chat_completions_api=False, **self.valid_config
        )

        # Build the request with a formatted prompt
        payload = client.build_request(formatted_prompt="Hello, how are you?", temperature=0.5)

        # Verify the payload
        self.assertIn("prompt", payload)
        self.assertEqual(payload["prompt"], "Hello, how are you?")
        self.assertEqual(payload["temperature"], 0.5)

    @patch("grasp.core.models.client.openai_azure_client.AzureOpenAI")
    def test_build_request_completions_invalid_prompt(self, mock_openai):
        """Test build_request with completions API and invalid prompt"""
        mock_openai.return_value = MagicMock()

        client = OpenAIAzureClient(
            async_client=False, chat_completions_api=False, **self.valid_config
        )

        # Try to build request with None prompt
        with self.assertRaises(ValueError) as context:
            client.build_request(formatted_prompt=None)

        # Verify the error message
        self.assertIn("formatted_prompt passed is None", str(context.exception))

    @patch("grasp.core.models.client.openai_azure_client.AzureOpenAI")
    def test_send_request_chat_completions(self, mock_openai):
        """Test send_request with chat completions API"""
        # Create a mock for the client and completions
        mock_chat_completions = MagicMock()
        mock_openai.return_value = MagicMock()
        mock_openai.return_value.chat.completions.create = mock_chat_completions

        client = OpenAIAzureClient(async_client=False, **self.valid_config)

        # Prepare payload
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello, how are you?"},
            ],
            "temperature": 0.7,
        }

        # Send the request
        client.send_request(payload, "gpt-4", {"stream": False})

        # Verify the request was sent correctly
        mock_chat_completions.assert_called_once_with(
            messages=payload["messages"], temperature=0.7, model="gpt-4", stream=False
        )

    @patch("grasp.core.models.client.openai_azure_client.AzureOpenAI")
    def test_send_request_completions(self, mock_openai):
        """Test send_request with completions API"""
        # Create a mock for the client and completions
        mock_completions = MagicMock()
        mock_openai.return_value = MagicMock()
        mock_openai.return_value.completions.create = mock_completions

        client = OpenAIAzureClient(
            async_client=False, chat_completions_api=False, **self.valid_config
        )

        # Prepare payload
        payload = {"prompt": "Hello, how are you?", "temperature": 0.5}

        # Send the request
        client.send_request(payload, "gpt-4", {"max_tokens": 100})

        # Verify the request was sent correctly
        mock_completions.assert_called_once_with(
            prompt="Hello, how are you?", temperature=0.5, model="gpt-4", max_tokens=100
        )

    @patch("grasp.core.models.client.openai_azure_client.AsyncAzureOpenAI")
    def test_send_request_async(self, mock_async_openai):
        """Test send_request with async client"""
        # Create a mock for the client and completions
        mock_chat_completions = MagicMock()
        mock_async_openai.return_value = MagicMock()
        mock_async_openai.return_value.chat.completions.create = mock_chat_completions

        client = OpenAIAzureClient(async_client=True, **self.valid_config)

        # Prepare payload
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello, how are you?"},
            ]
        }

        # Send the request
        client.send_request(payload, "gpt-4")

        # Verify the request was sent correctly
        mock_chat_completions.assert_called_once_with(messages=payload["messages"], model="gpt-4")

    def test_azure_client_config_validation(self):
        """Test AzureClientConfig validation"""
        # Valid config
        config = AzureClientConfig(
            azure_deployment="gpt-4",
            azure_endpoint="https://test-endpoint.openai.azure.com",
            api_version="2023-05-15",
            api_key="test-api-key",
        )

        # Verify default values are set correctly
        self.assertEqual(config.timeout, 120)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.default_headers, {"Connection": "close"})

        # Test with custom values
        config = AzureClientConfig(
            azure_deployment="gpt-4",
            azure_endpoint="https://test-endpoint.openai.azure.com",
            api_version="2023-05-15",
            api_key="test-api-key",
            timeout=60,
            max_retries=5,
            default_headers={"Custom": "Header"},
        )

        self.assertEqual(config.timeout, 60)
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.default_headers, {"Custom": "Header"})

        # Verify model_dump works correctly
        config_dict = config.model_dump()
        self.assertEqual(config_dict["azure_deployment"], "gpt-4")
        self.assertEqual(config_dict["azure_endpoint"], "https://test-endpoint.openai.azure.com")
        self.assertEqual(config_dict["api_version"], "2023-05-15")
        self.assertEqual(config_dict["api_key"], "test-api-key")
        self.assertEqual(config_dict["timeout"], 60)
        self.assertEqual(config_dict["max_retries"], 5)
        self.assertEqual(config_dict["default_headers"], {"Custom": "Header"})

    @patch("grasp.core.models.client.openai_azure_client.AzureOpenAI")
    def test_native_structured_output(self, mock_azure_openai):
        """Test structured output using OpenAI's beta.chat.completions.parse"""
        # Set up mock client
        mock_client = MagicMock()
        mock_azure_openai.return_value = mock_client

        # Set up mock beta.chat.completions.parse response
        structured_data = {
            "name": "John",
            "age": 30,
            "skills": ["python", "javascript"],
        }
        mock_parse = MagicMock(return_value=structured_data)
        mock_client.beta.chat.completions.parse = mock_parse

        # Create client instance
        client = OpenAIAzureClient(async_client=False, **self.valid_config)

        # Create messages and payload
        messages = [{"role": "user", "content": "Get user info"}]
        payload = client.build_request(messages=messages, temperature=0.7)

        # Call send_request with pydantic_model in generation_params
        response = client.send_request(
            payload=payload,
            model_name="gpt-4",
            generation_params={"pydantic_model": SampleStructuredOutput},
        )

        # Verify beta.chat.completions.parse was called with correct parameters
        mock_parse.assert_called_once_with(
            model="gpt-4",
            messages=payload["messages"],
            response_format=SampleStructuredOutput,
        )

        # Verify the structured data was returned correctly
        self.assertEqual(response, structured_data)

    @patch("grasp.core.models.client.openai_azure_client.AzureOpenAI")
    def test_pydantic_validation_error(self, mock_azure_openai):
        """Test handling of validation errors in structured output"""
        # Set up mock client
        mock_client = MagicMock()
        mock_azure_openai.return_value = mock_client

        # Set up mock beta.chat.completions.parse to raise ValueError
        mock_parse = MagicMock(side_effect=ValueError("Field 'age' required"))
        mock_client.beta.chat.completions.parse = mock_parse

        # Create client instance
        client = OpenAIAzureClient(async_client=False, **self.valid_config)

        # Create messages and payload
        messages = [{"role": "user", "content": "Get user info"}]
        payload = client.build_request(messages=messages)

        # Verify ValueError is raised when pydantic validation fails
        with self.assertRaises(ValueError):
            client.send_request(
                payload=payload,
                model_name="gpt-4",
                generation_params={"pydantic_model": SampleStructuredOutput},
            )

        # Verify beta.chat.completions.parse was called with correct parameters
        mock_parse.assert_called_once_with(
            model="gpt-4",
            messages=payload["messages"],
            response_format=SampleStructuredOutput,
        )

    @patch("grasp.core.models.client.openai_azure_client.AzureOpenAI")
    def test_json_decode_error(self, mock_azure_openai):
        """Test handling of JSON decode errors in structured output"""
        # Set up mock client
        mock_client = MagicMock()
        mock_azure_openai.return_value = mock_client

        # Set up mock beta.chat.completions.parse to raise JSONDecodeError
        mock_parse = MagicMock(side_effect=json.JSONDecodeError("Expecting property name", "", 0))
        mock_client.beta.chat.completions.parse = mock_parse

        # Create client instance
        client = OpenAIAzureClient(async_client=False, **self.valid_config)

        # Create messages and payload
        messages = [{"role": "user", "content": "Get user info"}]
        payload = client.build_request(messages=messages)

        # Verify JSONDecodeError is raised when JSON parsing fails
        with self.assertRaises(json.JSONDecodeError):
            client.send_request(
                payload=payload,
                model_name="gpt-4",
                generation_params={"pydantic_model": SampleStructuredOutput},
            )

        # Verify beta.chat.completions.parse was called with correct parameters
        mock_parse.assert_called_once_with(
            model="gpt-4",
            messages=payload["messages"],
            response_format=SampleStructuredOutput,
        )


if __name__ == "__main__":
    unittest.main()
