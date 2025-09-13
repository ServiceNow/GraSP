import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the parent directory to sys.path to import the necessary modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

import json

from grasp.core.models.client.http_client import HttpClient, HttpClientConfig


class TestHttpClientConfig(unittest.TestCase):
    """Unit tests for the HttpClientConfig class"""

    def test_config_defaults(self):
        """Test default values for HttpClientConfig"""
        config = HttpClientConfig(base_url="https://test-api.com")
        self.assertEqual(config.base_url, "https://test-api.com")
        self.assertEqual(config.headers, {})
        self.assertEqual(config.timeout, 120)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.ssl_verify, True)
        self.assertEqual(config.ssl_cert, None)

    def test_config_custom_values(self):
        """Test custom values for HttpClientConfig"""
        config = HttpClientConfig(
            base_url="https://custom-api.com",
            headers={"Authorization": "Bearer token123"},
            timeout=60,
            max_retries=5,
            ssl_verify=False,
            ssl_cert=None,
        )
        self.assertEqual(config.base_url, "https://custom-api.com")
        self.assertEqual(config.headers, {"Authorization": "Bearer token123"})
        self.assertEqual(config.timeout, 60)
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.ssl_verify, False)
        self.assertEqual(config.ssl_cert, None)


class TestHttpClient(unittest.TestCase):
    """Unit tests for the HttpClient class"""

    def setUp(self):
        """Set up test fixtures"""
        self.base_url = "https://test-api.com"
        self.headers = {"Content-Type": "application/json"}
        self.client = HttpClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=30,
            max_retries=2,
            verify_ssl=True,
            verify_cert=None,
        )

    def test_initialization(self):
        """Test client initialization with various parameters"""
        # Test with standard parameters
        self.assertEqual(self.client.base_url, self.base_url)
        self.assertEqual(self.client.headers, self.headers)
        self.assertEqual(self.client.timeout, 30)
        self.assertEqual(self.client.max_retries, 2)
        self.assertEqual(self.client.verify_ssl, True)
        self.assertIsNone(self.client.stop)

        # Test with stop sequences
        stop_sequences = ["###", "END"]
        client_with_stop = HttpClient(base_url=self.base_url, stop=stop_sequences)
        self.assertEqual(client_with_stop.stop, stop_sequences)

    def test_build_request(self):
        """Test build_request method constructs payloads correctly"""
        # Test basic payload
        base_payload = {"prompt": "Hello, world!"}
        result = self.client.build_request_with_payload(base_payload)
        self.assertEqual(result, {"prompt": "Hello, world!"})

        # Test with additional kwargs
        result = self.client.build_request_with_payload(
            base_payload, temperature=0.7, max_tokens=100
        )
        expected = {"prompt": "Hello, world!", "temperature": 0.7, "max_tokens": 100}
        self.assertEqual(result, expected)

        # Test with stop sequences
        client_with_stop = HttpClient(base_url=self.base_url, stop=["###", "END"])
        result = client_with_stop.build_request_with_payload(base_payload)
        expected = {
            "prompt": "Hello, world!",
            "stop": ["###", "END"],
            "temperature": 0.7,
            "max_tokens": 100,
        }
        self.assertEqual(result, expected)

    @patch("requests.request")
    def test_send_request(self, mock_request):
        """Test send_request method sends HTTP requests correctly"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = '{"result": "Success"}'
        mock_response.json.return_value = {"result": "Success"}
        mock_request.return_value = mock_response

        # Test basic request
        payload = {"prompt": "Test prompt"}
        response = self.client.send_request(payload)

        # Verify request was made with correct parameters
        mock_request.assert_called_once_with(
            "POST",
            self.base_url,
            headers=self.headers,
            data=json.dumps(payload).encode(),
            timeout=30,
            verify=True,
            cert=None,
        )
        self.assertEqual(response, mock_response)

        # Test with generation parameters
        mock_request.reset_mock()
        generation_params = {"temperature": 0.8, "max_tokens": 50}
        response = self.client.send_request(
            payload, model_name="test-model", generation_params=generation_params
        )

        expected_payload = {
            "prompt": "Test prompt",
            "temperature": 0.8,
            "max_tokens": 50,
        }
        mock_request.assert_called_once_with(
            "POST",
            self.base_url,
            headers=self.headers,
            data=json.dumps(expected_payload).encode(),
            timeout=30,
            verify=True,
            cert=None,
        )

    @patch("requests.request")
    def test_send_request_exception_handling(self, mock_request):
        """Test send_request handles exceptions correctly"""
        # Setup mock to raise exception
        mock_request.side_effect = Exception("Network error")

        # Test request with exception
        payload = {"prompt": "Test prompt"}
        response = self.client.send_request(payload)

        # Verify empty response is returned on exception
        self.assertEqual(response, "")

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_async_send_request(self, mock_post):
        """Test async_send_request method sends HTTP requests correctly"""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value='{"result": "Success"}')
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_post.return_value = mock_response

        # Test basic async request
        payload = {"prompt": "Test prompt"}
        response = await self.client.async_send_request(payload)

        # Verify request was made with correct parameters
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        self.assertEqual(call_kwargs["url"], self.base_url)
        self.assertEqual(call_kwargs["headers"], self.headers)
        self.assertEqual(call_kwargs["timeout"], 30)
        self.assertEqual(call_kwargs["ssl"], False)
        self.assertEqual(json.loads(call_kwargs["data"].decode()), payload)

        # Check that response is properly processed
        self.assertEqual(response.text, '{"result": "Success"}')
        self.assertEqual(response.status_code, 200)

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_async_send_request_with_generation_params(self, mock_post):
        """Test async_send_request with generation parameters"""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value='{"result": "Success"}')
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_post.return_value = mock_response

        # Test request with generation parameters
        payload = {"prompt": "Test prompt"}
        generation_params = {"temperature": 0.8, "max_tokens": 50}
        await self.client.async_send_request(
            payload, model_name="test-model", generation_params=generation_params
        )

        # Verify payload was updated with generation parameters
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        sent_payload = json.loads(call_kwargs["data"].decode())
        self.assertEqual(sent_payload["prompt"], "Test prompt")
        self.assertEqual(sent_payload["temperature"], 0.8)
        self.assertEqual(sent_payload["max_tokens"], 50)

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_async_send_request_exception_handling(self, mock_post):
        """Test async_send_request handles exceptions correctly"""
        # Setup mock to raise exception
        mock_post.side_effect = Exception("Network error")

        # Test request with exception
        payload = {"prompt": "Test prompt"}
        response = await self.client.async_send_request(payload)

        # Verify empty response is returned on exception
        self.assertEqual(response, "")


if __name__ == "__main__":
    unittest.main()
