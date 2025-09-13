import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path to import the necessary modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from grasp.core.models.client.client_factory import ClientFactory, ModelType


class TestClientFactory(unittest.TestCase):
    """Unit tests for the ClientFactory class"""

    def test_model_type_enum(self):
        """Test ModelType enum values are correct"""
        self.assertEqual(ModelType.VLLM.value, "vllm")
        self.assertEqual(ModelType.OPENAI.value, "azure_openai")
        self.assertEqual(ModelType.AZURE.value, "azure")
        self.assertEqual(ModelType.MISTRALAI.value, "mistralai")
        self.assertEqual(ModelType.TGI.value, "tgi")

    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_client_invalid_model_type(self, mock_validate):
        """Test create_client with invalid model type raises ValueError"""
        model_url = "http://test-url.com"
        model_auth_token = "test-token"
        model_config = {
            "model_type": "invalid_type",
            "url": model_url,
            "auth_token": model_auth_token,
        }
        with self.assertRaises(ValueError) as context:
            ClientFactory.create_client(model_config, model_url, model_auth_token)
        self.assertIn("Unsupported model type", str(context.exception))
        self.assertIn("Must be one of", str(context.exception))

    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_client_missing_model_type(self, mock_validate):
        """Test create_client with missing model_type key"""
        mock_validate.side_effect = ValueError("Missing required key: model_type")
        model_config = {"url": "http://test-url.com"}
        with self.assertRaises(ValueError):
            ClientFactory.create_client(model_config, "http://test-url.com")

    def test_create_client_missing_url(self):
        """Test create_client with missing model_type key"""
        model_config = {"url": "http://test-url.com", "model_type": "vllm"}
        with self.assertRaises(TypeError) as context:
            ClientFactory.create_client(model_config)
        self.assertIn("missing 1 required positional argument: 'url'", str(context.exception))

    def test_create_client_missing_auth_token(self):
        """Test create_client with missing model_type key"""
        model_url = "http://test-url.com"
        model_config = {
            "url": model_url,
            "auth_token": "test-token",
            "model_type": "vllm",
        }
        with self.assertRaises(ValueError) as context:
            ClientFactory.create_client(model_config, model_url)
        self.assertIn("Auth token/API key is required for client creation", str(context.exception))

    @patch("grasp.core.models.client.client_factory.OpenAIClient")
    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_openai_client(self, mock_validate, mock_openai_client):
        """Test _create_openai_client method"""
        # Setup mock
        mock_openai_client.return_value = MagicMock()
        model_url = "http://vllm-test.com"
        auth_token = "test-token"
        # Test with valid config
        model_config = {
            "model_type": "vllm",
            "url": model_url,
            "auth_token": auth_token,
            "timeout": 60,
            "max_retries": 5,
        }

        client = ClientFactory._create_openai_client(model_config, model_url, auth_token)

        # Verify the client was created with the right parameters
        mock_validate.assert_called_once_with(["url", "auth_token"], model_config, "model")
        self.assertIsNotNone(client)
        mock_openai_client.assert_called_once()

        # Check if parameters were passed correctly
        args, kwargs = mock_openai_client.call_args
        self.assertEqual(kwargs["base_url"], "http://vllm-test.com")
        self.assertEqual(kwargs["api_key"], "test-token")
        self.assertEqual(kwargs["timeout"], 60)
        self.assertEqual(kwargs["max_retries"], 5)

    @patch("grasp.core.models.client.client_factory.OpenAIClient")
    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_openai_client_multi_url(self, mock_validate, mock_openai_client):
        """Test _create_openai_client method"""
        # Setup mock
        mock_openai_client.return_value = MagicMock()
        model_url1 = "http://vllm-test1.com"
        model_url2 = "http://vllm-test2.com"
        auth_token1 = "test-token1"
        auth_token2 = "test-token2"

        # Test with valid config
        model_config = {
            "model_type": "vllm",
            "url": [model_url1, model_url2],
            "auth_token": [auth_token1, auth_token2],
            "timeout": 60,
            "max_retries": 5,
        }

        client = ClientFactory._create_openai_client(model_config, model_url1, auth_token1)

        # Verify the client was created with the right parameters
        mock_validate.assert_called_once_with(["url", "auth_token"], model_config, "model")
        self.assertIsNotNone(client)
        mock_openai_client.assert_called_once()

        # Check if parameters were passed correctly
        args, kwargs = mock_openai_client.call_args
        self.assertEqual(kwargs["base_url"], model_url1)
        self.assertEqual(kwargs["api_key"], auth_token1)
        self.assertEqual(kwargs["timeout"], 60)
        self.assertEqual(kwargs["max_retries"], 5)

    @patch("grasp.core.models.client.client_factory.OpenAIClient")
    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_openai_client_multi_url_single_auth_token(
        self, mock_validate, mock_openai_client
    ):
        """Test _create_openai_client method"""
        # Setup mock
        mock_openai_client.return_value = MagicMock()
        model_url1 = "http://vllm-test1.com"
        model_url2 = "http://vllm-test2.com"
        auth_token = "test-token1"

        # Test with valid config
        model_config = {
            "model_type": "vllm",
            "url": [model_url1, model_url2],
            "auth_token": auth_token,
            "timeout": 60,
            "max_retries": 5,
        }

        client = ClientFactory._create_openai_client(model_config, model_url1, auth_token)

        # Verify the client was created with the right parameters
        mock_validate.assert_called_once_with(["url", "auth_token"], model_config, "model")
        self.assertIsNotNone(client)
        mock_openai_client.assert_called_once()

        # Check if parameters were passed correctly
        args, kwargs = mock_openai_client.call_args
        self.assertEqual(kwargs["base_url"], model_url1)
        self.assertEqual(kwargs["api_key"], auth_token)
        self.assertEqual(kwargs["timeout"], 60)
        self.assertEqual(kwargs["max_retries"], 5)

    @patch("grasp.core.models.client.client_factory.OpenAIClient")
    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_openai_client_with_completions_api(self, mock_validate, mock_openai_client):
        """Test _create_openai_client method with completion API flag"""
        # Setup mock
        mock_openai_client.return_value = MagicMock()
        model_url = "http://vllm-test.com"
        auth_token = "test-token"
        # Test with valid config and completions_api set to True
        model_config = {
            "model_type": "vllm",
            "url": model_url,
            "auth_token": auth_token,
            "completions_api": True,
        }

        ClientFactory._create_openai_client(
            model_config,
            model_url,
            auth_token,
            async_client=True,
            chat_completions_api=False,
        )

        # Verify the client was created with the right parameters
        mock_validate.assert_called_once_with(["url", "auth_token"], model_config, "model")
        mock_openai_client.assert_called_once_with(True, False, **mock_openai_client.call_args[1])

    @patch("grasp.core.models.client.client_factory.OpenAIAzureClient")
    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_openai_azure_client(self, mock_validate, mock_openai_client):
        """Test _create_openai_client method"""
        # Setup mock
        mock_openai_client.return_value = MagicMock()

        model_url = "http://openai-test.com"
        auth_token = "test-key"

        # Test with valid config
        model_config = {
            "model_type": "azure_openai",
            "url": model_url,
            "api_key": auth_token,
            "api_version": "2023-05-15",
            "model": "gpt-4",
            "timeout": 90,
            "max_retries": 2,
        }

        client = ClientFactory._create_openai_azure_client(model_config, model_url, auth_token)

        # Verify the client was created with the right parameters
        mock_validate.assert_called_once_with(
            ["url", "api_key", "api_version", "model"], model_config, "model"
        )
        self.assertIsNotNone(client)
        mock_openai_client.assert_called_once()

        # Check if parameters were passed correctly
        args, kwargs = mock_openai_client.call_args
        self.assertEqual(kwargs["azure_endpoint"], model_url)
        self.assertEqual(kwargs["api_key"], auth_token)
        self.assertEqual(kwargs["api_version"], "2023-05-15")
        self.assertEqual(kwargs["azure_deployment"], "gpt-4")
        self.assertEqual(kwargs["timeout"], 90)
        self.assertEqual(kwargs["max_retries"], 2)

    @patch("grasp.core.models.client.client_factory.OpenAIAzureClient")
    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_openai_azure_client_multi_url(self, mock_validate, mock_openai_client):
        """Test _create_openai_client method"""
        # Setup mock
        mock_openai_client.return_value = MagicMock()

        model_url1 = "http://openai-test1.com"
        model_url2 = "http://openai-test2.com"
        auth_token1 = "test-key1"
        auth_token2 = "test-key2"

        # Test with valid config
        model_config = {
            "model_type": "azure_openai",
            "url": [model_url1, model_url2],
            "api_key": [auth_token1, auth_token2],
            "api_version": "2023-05-15",
            "model": "gpt-4",
            "timeout": 90,
            "max_retries": 2,
        }

        client = ClientFactory._create_openai_azure_client(model_config, model_url1, auth_token1)

        # Verify the client was created with the right parameters
        mock_validate.assert_called_once_with(
            ["url", "api_key", "api_version", "model"], model_config, "model"
        )
        self.assertIsNotNone(client)
        mock_openai_client.assert_called_once()

        # Check if parameters were passed correctly
        args, kwargs = mock_openai_client.call_args
        self.assertEqual(kwargs["azure_endpoint"], model_url1)
        self.assertEqual(kwargs["api_key"], auth_token1)
        self.assertEqual(kwargs["api_version"], "2023-05-15")
        self.assertEqual(kwargs["azure_deployment"], "gpt-4")
        self.assertEqual(kwargs["timeout"], 90)
        self.assertEqual(kwargs["max_retries"], 2)

    @patch("grasp.core.models.client.client_factory.OpenAIAzureClient")
    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_openai_azure_client_multi_url_single_auth_token(
        self, mock_validate, mock_openai_client
    ):
        """Test _create_openai_client method"""
        # Setup mock
        mock_openai_client.return_value = MagicMock()

        model_url1 = "http://openai-test1.com"
        model_url2 = "http://openai-test2.com"
        auth_token = "test-key"

        # Test with valid config
        model_config = {
            "model_type": "azure_openai",
            "url": [model_url1, model_url2],
            "api_key": auth_token,
            "api_version": "2023-05-15",
            "model": "gpt-4",
            "timeout": 90,
            "max_retries": 2,
        }

        client = ClientFactory._create_openai_azure_client(model_config, model_url1, auth_token)

        # Verify the client was created with the right parameters
        mock_validate.assert_called_once_with(
            ["url", "api_key", "api_version", "model"], model_config, "model"
        )
        self.assertIsNotNone(client)
        mock_openai_client.assert_called_once()

        # Check if parameters were passed correctly
        args, kwargs = mock_openai_client.call_args
        self.assertEqual(kwargs["azure_endpoint"], model_url1)
        self.assertEqual(kwargs["api_key"], auth_token)
        self.assertEqual(kwargs["api_version"], "2023-05-15")
        self.assertEqual(kwargs["azure_deployment"], "gpt-4")
        self.assertEqual(kwargs["timeout"], 90)
        self.assertEqual(kwargs["max_retries"], 2)

    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_azure_client(self, mock_validate):
        """Test _create_azure_client method"""
        # Test with valid config
        model_url = "http://azure-test.com"
        model_auth_token = "test-token"
        model_config = {
            "model_type": "azure",
            "url": model_url,
            "auth_token": model_auth_token,
            "timeout": 75,
        }

        http_client = ClientFactory._create_http_client(model_config, model_url, model_auth_token)

        # Verify the client config was created with the right parameters
        mock_validate.assert_called_once_with(["url", "auth_token"], model_config, "model")
        self.assertIsNotNone(http_client)
        self.assertEqual(http_client.headers["Authorization"], "Bearer test-token")
        self.assertEqual(http_client.headers["Content-Type"], "application/json")
        self.assertEqual(http_client.timeout, 75)

    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_azure_client_with_bearer_token(self, mock_validate):
        """Test _create_azure_client method with token that already has Bearer prefix"""
        # Test with valid config where token already has Bearer prefix
        model_url = "http://azure-test.com"
        model_auth_token = "Bearer test-token"
        model_config = {
            "model_type": "azure",
            "url": model_url,
            "auth_token": model_auth_token,
            "timeout": 75,
        }

        http_client = ClientFactory._create_http_client(model_config, model_url, model_auth_token)

        # Verify the client config was created with the right parameters
        self.assertEqual(http_client.headers["Authorization"], "Bearer test-token")

    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_azure_client_with_multi_url(self, mock_validate):
        """Test _create_azure_client method with token that already has Bearer prefix"""
        # Test with valid config where token already has Bearer prefix
        model_url1 = "http://azure-test1.com"
        model_auth_token1 = "Bearer test-token1"
        model_url2 = "http://azure-test2.com"
        model_auth_token2 = "Bearer test-token2"
        model_config = {
            "model_type": "azure",
            "url": [model_url1, model_url2],
            "auth_token": [model_auth_token1, model_auth_token2],
            "timeout": 75,
        }

        http_client = ClientFactory._create_http_client(model_config, model_url1, model_auth_token1)

        # Verify the client config was created with the right parameters
        self.assertEqual(http_client.headers["Authorization"], model_auth_token1)

    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_azure_client_with_multi_url_single_auth_token(self, mock_validate):
        """Test _create_azure_client method with token that already has Bearer prefix"""
        # Test with valid config where token already has Bearer prefix
        model_url1 = "http://azure-test1.com"
        model_url2 = "http://azure-test2.com"
        model_auth_token = "Bearer test-token"
        model_config = {
            "model_type": "azure",
            "url": [model_url1, model_url2],
            "auth_token": model_auth_token,
            "timeout": 75,
        }

        http_client = ClientFactory._create_http_client(model_config, model_url1, model_auth_token)

        # Verify the client config was created with the right parameters
        self.assertEqual(http_client.headers["Authorization"], model_auth_token)

    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_azure_client_with_token_list(self, mock_validate):
        """Test _create_azure_client method with token provided as a list"""
        # Test with valid config where token is provided as a list
        model_url = "http://azure-test.com"
        model_auth_token = "test-token"
        model_config = {
            "model_type": "azure",
            "url": model_url,
            "auth_token": [model_auth_token],
            "timeout": 75,
        }

        http_client = ClientFactory._create_http_client(model_config, model_url, model_auth_token)

        # Verify the client config was created with the right parameters
        self.assertEqual(http_client.headers["Authorization"], "Bearer test-token")

    @patch("grasp.core.models.client.client_factory.MistralAzure")
    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_mistral_client(self, mock_validate, mock_mistral_client):
        """Test _create_mistral_client method"""
        # Setup mock
        mock_mistral_client.return_value = MagicMock()
        model_url = "http://mistral-test.com"
        model_auth_token = "test-token"

        # Test with valid config
        model_config = {
            "model_type": "mistralai",
            "url": model_url,
            "auth_token": model_auth_token,
            "timeout": 120,
        }

        client = ClientFactory._create_mistral_client(
            model_config, model_url, model_auth_token, async_client=True
        )

        # Verify the client was created with the right parameters
        mock_validate.assert_called_once_with(["url", "auth_token"], model_config, "model")
        self.assertIsNotNone(client)
        mock_mistral_client.assert_called_once()

        # Check if parameters were passed correctly
        args, kwargs = mock_mistral_client.call_args
        self.assertEqual(kwargs["azure_endpoint"], model_url)
        self.assertEqual(kwargs["azure_api_key"], model_auth_token)
        self.assertIn("retry_config", kwargs)
        self.assertIn("async_client", kwargs)  # Should have the async client parameter

    @patch("grasp.core.models.client.client_factory.MistralAzure")
    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_mistral_client_multi_url(self, mock_validate, mock_mistral_client):
        """Test _create_mistral_client method"""
        # Setup mock
        mock_mistral_client.return_value = MagicMock()
        model_url1 = "http://mistral-test1.com"
        model_url2 = "http://mistral-test2.com"
        model_auth_token1 = "test-token1"
        model_auth_token2 = "test-token2"
        model_config = {
            "model_type": "mistralai",
            "url": [model_url1, model_url2],
            "auth_token": [model_auth_token1, model_auth_token2],
            "timeout": 120,
        }

        client = ClientFactory._create_mistral_client(
            model_config, model_url1, model_auth_token1, async_client=True
        )

        # Verify the client was created with the right parameters
        mock_validate.assert_called_once_with(["url", "auth_token"], model_config, "model")
        self.assertIsNotNone(client)
        mock_mistral_client.assert_called_once()

        # Check if parameters were passed correctly
        args, kwargs = mock_mistral_client.call_args
        self.assertEqual(kwargs["azure_endpoint"], model_url1)
        self.assertEqual(kwargs["azure_api_key"], model_auth_token1)
        self.assertIn("retry_config", kwargs)
        self.assertIn("async_client", kwargs)  # Should have the async client parameter

    @patch("grasp.core.models.client.client_factory.MistralAzure")
    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_mistral_client_multi_url_single_auth_token(
        self, mock_validate, mock_mistral_client
    ):
        """Test _create_mistral_client method"""
        # Setup mock
        mock_mistral_client.return_value = MagicMock()
        model_url1 = "http://mistral-test1.com"
        model_url2 = "http://mistral-test2.com"
        model_auth_token = "test-token"
        model_config = {
            "model_type": "mistralai",
            "url": [model_url1, model_url2],
            "auth_token": model_auth_token,
            "timeout": 120,
        }

        client = ClientFactory._create_mistral_client(
            model_config, model_url1, model_auth_token, async_client=True
        )

        # Verify the client was created with the right parameters
        mock_validate.assert_called_once_with(["url", "auth_token"], model_config, "model")
        self.assertIsNotNone(client)
        mock_mistral_client.assert_called_once()

        # Check if parameters were passed correctly
        args, kwargs = mock_mistral_client.call_args
        self.assertEqual(kwargs["azure_endpoint"], model_url1)
        self.assertEqual(kwargs["azure_api_key"], model_auth_token)
        self.assertIn("retry_config", kwargs)
        self.assertIn("async_client", kwargs)  # Should have the async client parameter

    @patch("grasp.core.models.client.client_factory.MistralAzure")
    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_mistral_client_sync(self, mock_validate, mock_mistral_client):
        """Test _create_mistral_client method with synchronous client"""
        # Setup mock
        mock_mistral_client.return_value = MagicMock()
        model_url = "http://mistral-test1.com"
        model_auth_token = "test-token"
        # Test with valid config
        model_config = {
            "model_type": "mistralai",
            "url": model_url,
            "auth_token": model_auth_token,
            "timeout": 120,
        }

        client = ClientFactory._create_mistral_client(
            model_config, model_url, model_auth_token, async_client=False
        )

        # Verify the client was created with the right parameters
        mock_validate.assert_called_once_with(["url", "auth_token"], model_config, "model")
        self.assertIsNotNone(client)
        mock_mistral_client.assert_called_once()

        # Check if parameters were passed correctly
        args, kwargs = mock_mistral_client.call_args
        self.assertIn("client", kwargs)  # Should have the sync client parameter
        self.assertNotIn("async_client", kwargs)

    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_tgi_client(self, mock_validate):
        """Test _create_tgi_client method"""
        # Test with valid config
        model_url = "http://tgi-test.com"
        auth_token = "Bearer test-token"
        model_config = {
            "model_type": "tgi",
            "url": model_url,
            "auth_token": auth_token,
            "timeout": 180,
        }

        http_client = ClientFactory._create_http_client(model_config, model_url, auth_token)

        # Verify the client config was created with the right parameters
        mock_validate.assert_called_once_with(["url", "auth_token"], model_config, "model")
        self.assertIsNotNone(http_client)
        self.assertEqual(http_client.headers["Authorization"], auth_token)
        self.assertEqual(http_client.headers["Content-Type"], "application/json")
        self.assertEqual(http_client.timeout, 180)

    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_tgi_client_multi_url(self, mock_validate):
        """Test _create_tgi_client method"""
        # Test with valid config
        model_url1 = "http://tgi-test1.com"
        auth_token1 = "Bearer test-token1"
        model_url2 = "http://tgi-test2.com"
        auth_token2 = "Bearer test-token2"
        model_config = {
            "model_type": "tgi",
            "url": [model_url1, model_url2],
            "auth_token": [auth_token1, auth_token2],
            "timeout": 180,
        }

        http_client = ClientFactory._create_http_client(model_config, model_url1, auth_token1)

        # Verify the client config was created with the right parameters
        mock_validate.assert_called_once_with(["url", "auth_token"], model_config, "model")
        self.assertIsNotNone(http_client)
        self.assertEqual(http_client.headers["Authorization"], auth_token1)
        self.assertEqual(http_client.headers["Content-Type"], "application/json")
        self.assertEqual(http_client.timeout, 180)

    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_tgi_client_multi_url_single_auth_token(self, mock_validate):
        """Test _create_tgi_client method"""
        # Test with valid config
        model_url1 = "http://tgi-test1.com"
        model_url2 = "http://tgi-test2.com"
        auth_token = "Bearer test-token2"
        model_config = {
            "model_type": "tgi",
            "url": [model_url1, model_url2],
            "auth_token": auth_token,
            "timeout": 180,
        }

        http_client = ClientFactory._create_http_client(model_config, model_url1, auth_token)

        # Verify the client config was created with the right parameters
        mock_validate.assert_called_once_with(["url", "auth_token"], model_config, "model")
        self.assertIsNotNone(http_client)
        self.assertEqual(http_client.headers["Authorization"], auth_token)
        self.assertEqual(http_client.headers["Content-Type"], "application/json")
        self.assertEqual(http_client.timeout, 180)

    @patch("grasp.core.models.client.client_factory.utils.validate_required_keys")
    def test_create_tgi_client_with_bearer_token(self, mock_validate):
        """Test _create_tgi_client method with token that already has Bearer prefix"""
        # Test with valid config where token already has Bearer prefix
        model_url = "http://tgi-test.com"
        model_auth_token = "Bearer test-token"
        model_config = {
            "model_type": "tgi",
            "url": model_url,
            "auth_token": model_auth_token,
            "timeout": 180,
        }

        http_client = ClientFactory._create_http_client(model_config, model_url, model_auth_token)

        # Verify the client config was created with the right parameters
        self.assertEqual(http_client.headers["Authorization"], model_auth_token)

    @patch("grasp.core.models.client.client_factory.ClientFactory._create_openai_client")
    def test_create_client_openai(self, mock_create_vllm):
        """Test create_client with VLLM model type"""
        mock_create_vllm.return_value = MagicMock()
        model_url = "http://vllm-test.com"
        model_auth_token = "test-token"

        model_config = {
            "model_type": "vllm",
            "url": model_url,
            "auth_token": model_auth_token,
        }
        ClientFactory.create_client(model_config, model_url, model_auth_token, async_client=True)

        mock_create_vllm.assert_called_once_with(
            model_config, model_url, model_auth_token, True, True
        )

    @patch("grasp.core.models.client.client_factory.ClientFactory._create_openai_azure_client")
    def test_create_client_openai_azure(self, mock_create_openai):
        """Test create_client with OpenAI model type"""
        mock_create_openai.return_value = MagicMock()
        model_url = "http://openai-test.com"
        model_auth_token = "test-token"

        model_config = {
            "model_type": "azure_openai",
            "url": model_url,
            "auth_token": model_auth_token,
        }
        ClientFactory.create_client(model_config, model_url, model_auth_token, async_client=True)

        mock_create_openai.assert_called_once_with(
            model_config, model_url, model_auth_token, True, True
        )

    @patch("grasp.core.models.client.client_factory.ClientFactory._create_http_client")
    def test_create_client_azure(self, mock_create_http):
        """Test create_client with Azure model type"""
        mock_create_http.return_value = {"headers": {}, "timeout": 120}
        model_url = "http://azure-test.com"
        model_auth_token = "test-token"

        model_config = {
            "model_type": "azure",
            "url": model_url,
            "auth_token": model_auth_token,
        }
        result = ClientFactory.create_client(model_config, model_url, model_auth_token)

        mock_create_http.assert_called_once_with(model_config, model_url, model_auth_token)
        self.assertEqual(result, {"headers": {}, "timeout": 120})

    @patch("grasp.core.models.client.client_factory.ClientFactory._create_http_client")
    def test_create_client_triton(self, mock_create_http):
        """Test create_client with Triton model type"""
        mock_create_http.return_value = {"headers": {}, "timeout": 120}
        model_url = "http://triton-test.com"
        model_auth_token = "test-token"

        model_config = {
            "model_type": "triton",
            "url": model_url,
            "auth_token": model_auth_token,
        }
        result = ClientFactory.create_client(model_config, model_url, model_auth_token)

        mock_create_http.assert_called_once_with(model_config, model_url, model_auth_token)
        self.assertEqual(result, {"headers": {}, "timeout": 120})

    @patch("grasp.core.models.client.client_factory.ClientFactory._create_mistral_client")
    def test_create_client_mistralai(self, mock_create_mistral):
        """Test create_client with MistralAI model type"""
        mock_create_mistral.return_value = MagicMock()
        model_url = "http://mistral-test.com"
        model_auth_token = "test-token"

        model_config = {
            "model_type": "mistralai",
            "url": model_url,
            "auth_token": model_auth_token,
        }
        ClientFactory.create_client(model_config, model_url, model_auth_token, async_client=True)

        mock_create_mistral.assert_called_once_with(model_config, model_url, model_auth_token, True)

    @patch("grasp.core.models.client.client_factory.ClientFactory._create_http_client")
    def test_create_client_tgi(self, mock_create_http):
        """Test create_client with TGI model type"""
        mock_create_http.return_value = {"headers": {}, "timeout": 120}
        model_url = "http://tgi-test.com"
        model_auth_token = "test-token"

        model_config = {
            "model_type": "tgi",
            "url": model_url,
            "auth_token": model_auth_token,
        }
        result = ClientFactory.create_client(model_config, model_url, model_auth_token)

        mock_create_http.assert_called_once_with(model_config, model_url, model_auth_token)
        self.assertEqual(result, {"headers": {}, "timeout": 120})

    @patch("grasp.core.models.client.client_factory.ClientFactory._create_openai_client")
    def test_create_client_completions_api(self, mock_create_openai):
        """Test create_client with completions_api flag"""
        mock_create_openai.return_value = MagicMock()
        model_url = "http://vllm-test.com"
        model_auth_token = "test-token"

        # Test with completions_api=True
        model_config = {
            "model_type": "vllm",
            "url": model_url,
            "auth_token": model_auth_token,
            "completions_api": True,
        }
        ClientFactory.create_client(model_config, model_url, model_auth_token, async_client=True)

        mock_create_openai.assert_called_once_with(
            model_config, model_url, model_auth_token, True, False
        )


if __name__ == "__main__":
    unittest.main()
