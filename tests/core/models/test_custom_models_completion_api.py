import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add the parent directory to sys.path to import the necessary modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import sygra.utils.constants as constants
from sygra.core.models.custom_models import (
    BaseCustomModel,
    CustomOllama,
    CustomOpenAI,
    CustomVLLM,
)


class TestValidateCompletionApiSupport(unittest.TestCase):
    """Unit tests for the _validate_completions_api_support method in custom models"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Store original COMPLETION_ONLY_MODELS to restore after tests
        self.original_completion_only_models = (
            constants.COMPLETION_ONLY_MODELS.copy()
            if hasattr(constants, "COMPLETION_ONLY_MODELS")
            else []
        )

        # Common model config for testing
        self.base_config = {
            "name": "test_model",
            "parameters": {"temperature": 0.7, "max_tokens": 100},
        }

        # Config for CustomVLLM
        self.vllm_config = {
            **self.base_config,
            "name": "test_vllm",
            "model_type": "vllm",
            "url": "http://vllm-test.com",
            "auth_token": "test-token",
        }

        # Config for CustomOpenAI
        self.openai_config = {
            **self.base_config,
            "name": "test_openai",
            "model_type": "azure_openai",
            "url": "http://openai-test.com",
            "auth_token": "test-key",
            "api_version": "2023-05-15",
            "model": "gpt-4",
        }

        # Config for CustomOllama
        self.ollama_config = {
            **self.base_config,
            "name": "test_ollama",
            "model_type": "ollama",
        }

    def tearDown(self):
        """Clean up after each test method"""
        # Restore original COMPLETION_ONLY_MODELS
        constants.COMPLETION_ONLY_MODELS = self.original_completion_only_models

    @patch("sygra.core.models.custom_models.ClientFactory")
    @patch("sygra.core.models.custom_models.logger")
    def test_base_model_completions_api_not_supported(self, mock_logger, mock_client_factory):
        """Test that BaseCustomModel raises an error when completions_api is set to True"""

        # Create a test implementation of BaseCustomModel
        class TestBaseModel(BaseCustomModel):
            def _generate_response(self, *args, **kwargs):
                pass

            def name(self):
                return self.model_config.get("name")

        # Test with completions_api set to True
        config = {**self.base_config, "completions_api": True}

        # BaseCustomModel should raise an error if completions_api is True
        with self.assertRaises(ValueError) as context:
            TestBaseModel(config)

        # Verify the error message
        self.assertIn("does not support completion API", str(context.exception))
        self.assertIn(self.base_config["name"], str(context.exception))
        self.assertIn("models.yaml", str(context.exception))

    @patch("sygra.core.models.custom_models.ClientFactory")
    @patch("sygra.core.models.custom_models.logger")
    def test_base_model_completions_api_not_set(self, mock_logger, mock_client_factory):
        """Test that BaseCustomModel doesn't raise an error when completions_api is not set"""

        # Create a test implementation of BaseCustomModel
        class TestBaseModel(BaseCustomModel):
            def _generate_response(self, *args, **kwargs):
                pass

            def name(self):
                return self.model_config.get("name")

        # Test with completions_api not set
        try:
            TestBaseModel(self.base_config)
        except ValueError as e:
            self.fail(f"BaseCustomModel raised ValueError unexpectedly: {e}")

    @patch("sygra.core.models.custom_models.ClientFactory")
    @patch("sygra.core.models.custom_models.logger")
    def test_vllm_model_completions_api_supported(self, mock_logger, mock_client_factory):
        """Test that CustomVLLM supports completion API"""
        # Test with completions_api set to True
        config = {**self.vllm_config, "completions_api": True}

        # Create the model - should not raise an error
        try:
            CustomVLLM(config)
        except ValueError:
            self.fail("CustomVLLM raised ValueError unexpectedly")

        # Verify that logger.info was called with the appropriate message
        mock_logger.info.assert_called_once_with(f"Model {config['name']} supports completion API.")

    @patch("sygra.core.models.custom_models.ClientFactory")
    @patch("sygra.core.models.custom_models.logger")
    def test_vllm_model_completions_api_not_set(self, mock_logger, mock_client_factory):
        """Test that CustomVLLM doesn't log anything when completions_api is not set"""
        # Create the model with completions_api not set
        CustomVLLM(self.vllm_config)

        # Verify that logger.info was not called
        mock_logger.info.assert_not_called()

    @patch("sygra.core.models.custom_models.ClientFactory")
    @patch("sygra.core.models.custom_models.logger")
    def test_openai_model_completions_api_supported(self, mock_logger, mock_client_factory):
        """Test that CustomOpenAI supports completion API"""
        # Test with completions_api set to True
        config = {**self.openai_config, "completions_api": True}

        # Create the model - should not raise an error
        try:
            CustomOpenAI(config)
        except ValueError:
            self.fail("CustomOpenAI raised ValueError unexpectedly")

        # Verify that logger.info was called with the appropriate message
        mock_logger.info.assert_called_once_with(f"Model {config['name']} supports completion API.")

    @patch("sygra.core.models.custom_models.ClientFactory")
    @patch("sygra.core.models.custom_models.logger")
    def test_openai_model_completions_api_not_set(self, mock_logger, mock_client_factory):
        """Test that CustomOpenAI doesn't log anything when completions_api is not set"""
        # Create the model with completions_api not set
        CustomOpenAI(self.openai_config)

        # Verify that logger.info was not called
        mock_logger.info.assert_not_called()

    @patch("sygra.core.models.custom_models.logger")
    def test_ollama_model_completions_api_supported(self, mock_logger):
        """Test that CustomOllama supports completion API"""
        # Test with completions_api set to True
        config = {**self.ollama_config, "completions_api": True}

        # Create the model - should not raise an error
        try:
            CustomOllama(config)
        except ValueError:
            self.fail("CustomOllama raised ValueError unexpectedly")

        # Verify that logger.info was called with the appropriate message
        mock_logger.info.assert_called_once_with(f"Model {config['name']} supports completion API.")

    @patch("sygra.core.models.custom_models.logger")
    def test_ollama_model_completions_api_not_set(self, mock_logger):
        """Test that CustomOllama doesn't log anything when completions_api is not set"""
        # Create the model with completions_api not set
        CustomOllama(self.ollama_config)

        # Verify that logger.info was not called
        mock_logger.info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
