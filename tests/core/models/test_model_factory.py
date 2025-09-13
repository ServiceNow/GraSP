import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path to import the necessary modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from grasp.core.models.custom_models import (
    CustomAzure,
    CustomMistralAPI,
    CustomOllama,
    CustomOpenAI,
    CustomTGI,
    CustomTriton,
    CustomVLLM,
)
from grasp.core.models.langgraph.openai_chat_model import CustomOpenAIChatModel
from grasp.core.models.langgraph.vllm_chat_model import CustomVLLMChatModel
from grasp.core.models.model_factory import ModelFactory


class TestModelFactory(unittest.TestCase):
    """Unit tests for the ModelFactory class"""

    @patch("grasp.utils.utils.load_model_config")
    def test_update_model_config(self, mock_load_model_config):
        """Test _update_model_config static method"""
        # Mock the global model config
        mock_load_model_config.return_value = {
            "test_model": {
                "model_type": "vllm",
                "parameters": {"temperature": 0.7, "max_tokens": 100},
                "url": "http://test-url.com",
            }
        }

        # Test model config update
        model_config = {"name": "test_model", "parameters": {"temperature": 0.8}}
        updated_config = ModelFactory._update_model_config(model_config)

        # Verify the config was updated correctly
        self.assertEqual(updated_config["model_type"], "vllm")
        self.assertEqual(
            updated_config["parameters"]["temperature"], 0.8
        )  # Should use the provided value
        self.assertEqual(
            updated_config["parameters"]["max_tokens"], 100
        )  # Should retain the global value
        self.assertEqual(
            updated_config["url"], "http://test-url.com"
        )  # Should retain the global value

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.validate_required_keys")
    def test_update_model_config_with_nested_dict(self, mock_validate, mock_load_model_config):
        """Test _update_model_config with nested dictionary values"""
        # Mock the global model config
        mock_load_model_config.return_value = {
            "test_model": {
                "model_type": "vllm",
                "parameters": {"temperature": 0.7, "max_tokens": 100},
                "additional_params": {"param1": "value1", "param2": "value2"},
            }
        }

        # Test model config update with nested dict
        model_config = {
            "name": "test_model",
            "parameters": {"temperature": 0.8},
            "additional_params": {"param1": "new_value"},
        }

        updated_config = ModelFactory._update_model_config(model_config)

        # Verify nested dictionary was updated correctly
        self.assertEqual(updated_config["additional_params"]["param1"], "new_value")
        self.assertEqual(updated_config["additional_params"]["param2"], "value2")

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.validate_required_keys")
    def test_create_model_vllm(self, mock_validate, mock_load_model_config):
        """Test create_model with VLLM model type"""
        mock_load_model_config.return_value = {
            "test_vllm": {
                "model_type": "vllm",
                "url": "http://vllm-test.com",
                "auth_token": "test-token",
                "parameters": {},
            }
        }

        with patch.object(CustomVLLM, "__init__", return_value=None) as mock_init:
            model_config = {"name": "test_vllm", "model_type": "vllm"}
            ModelFactory.create_model(model_config)
            mock_init.assert_called_once()

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.validate_required_keys")
    def test_create_model_mistralai(self, mock_validate, mock_load_model_config):
        """Test create_model with MistralAI model type"""
        mock_load_model_config.return_value = {
            "test_mistral": {
                "model_type": "mistralai",
                "url": "http://mistral-test.com",
                "auth_token": "test-token",
                "parameters": {},
            }
        }

        with patch.object(CustomMistralAPI, "__init__", return_value=None) as mock_init:
            model_config = {"name": "test_mistral", "model_type": "mistralai"}
            ModelFactory.create_model(model_config)
            mock_init.assert_called_once()

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.validate_required_keys")
    def test_create_model_tgi(self, mock_validate, mock_load_model_config):
        """Test create_model with TGI model type"""
        mock_load_model_config.return_value = {
            "test_tgi": {
                "model_type": "tgi",
                "url": "http://tgi-test.com",
                "auth_token": "test-token",
                "parameters": {},
            }
        }

        with patch.object(CustomTGI, "__init__", return_value=None) as mock_init:
            model_config = {"name": "test_tgi", "model_type": "tgi"}
            ModelFactory.create_model(model_config)
            mock_init.assert_called_once()

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.validate_required_keys")
    def test_create_model_azure(self, mock_validate, mock_load_model_config):
        """Test create_model with Azure model type"""
        mock_load_model_config.return_value = {
            "test_azure": {
                "model_type": "azure",
                "url": "http://azure-test.com",
                "auth_token": "test-token",
                "parameters": {},
            }
        }

        with patch.object(CustomAzure, "__init__", return_value=None) as mock_init:
            model_config = {"name": "test_azure", "model_type": "azure"}
            ModelFactory.create_model(model_config)
            mock_init.assert_called_once()

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.validate_required_keys")
    def test_create_model_openai(self, mock_validate, mock_load_model_config):
        """Test create_model with OpenAI model type"""
        mock_load_model_config.return_value = {
            "test_openai": {
                "model_type": "azure_openai",
                "url": "http://openai-test.com",
                "api_key": "test-key",
                "api_version": "2023-05-15",
                "model": "gpt-4",
                "parameters": {},
            }
        }

        with patch.object(CustomOpenAI, "__init__", return_value=None) as mock_init:
            model_config = {"name": "test_openai", "model_type": "azure_openai"}
            ModelFactory.create_model(model_config)
            mock_init.assert_called_once()

    @patch("grasp.utils.utils.load_model_config")
    def test_create_model_ollama(self, mock_load_model_config):
        """Test create_model with Ollama model type"""
        mock_load_model_config.return_value = {
            "test_ollama": {"model_type": "ollama", "parameters": {}}
        }

        with patch.object(CustomOllama, "__init__", return_value=None) as mock_init:
            model_config = {"name": "test_ollama", "model_type": "ollama"}
            ModelFactory.create_model(model_config)
            mock_init.assert_called_once()

    @patch("grasp.utils.utils.load_model_config")
    def test_create_model_triton(self, mock_load_model_config):
        """Test create_model with Triton model type"""
        mock_load_model_config.return_value = {
            "test_triton": {
                "model_type": "triton",
                "url": "http://triton-test.com",
                "auth_token": "test-token",
                "payload_key": "default",
                "parameters": {},
            }
        }

        with patch.object(CustomTriton, "__init__", return_value=None) as mock_init:
            model_config = {"name": "test_triton", "model_type": "triton"}
            ModelFactory.create_model(model_config)
            mock_init.assert_called_once()

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.validate_required_keys")
    def test_create_model_langgraph_vllm(self, mock_validate, mock_load_model_config):
        """Test create_model with VLLM model type for langgraph backend"""
        mock_load_model_config.return_value = {
            "test_vllm": {
                "model_type": "vllm",
                "url": "http://vllm-test.com",
                "auth_token": "test-token",
                "parameters": {},
            }
        }

        with patch.object(CustomVLLMChatModel, "__init__", return_value=None) as mock_init:
            model_config = {"name": "test_vllm", "model_type": "vllm"}
            ModelFactory.create_model(model_config, "langgraph")
            mock_init.assert_called_once()

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.validate_required_keys")
    def test_create_model_langgraph_openai(self, mock_validate, mock_load_model_config):
        """Test create_model with OpenAI model type for langgraph backend"""
        mock_load_model_config.return_value = {
            "test_openai": {
                "model_type": "azure_openai",
                "url": "http://openai-test.com",
                "api_key": "test-key",
                "api_version": "2023-05-15",
                "model": "gpt-4",
                "parameters": {},
            }
        }

        with patch.object(CustomOpenAIChatModel, "__init__", return_value=None) as mock_init:
            model_config = {"name": "test_openai", "model_type": "azure_openai"}
            ModelFactory.create_model(model_config, "langgraph")
            mock_init.assert_called_once()

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.validate_required_keys")
    def test_create_model_unsupported_type(self, mock_validate, mock_load_model_config):
        """Test create_model with unsupported model type"""
        mock_load_model_config.return_value = {
            "test_unsupported": {
                "model_type": "unsupported",
                "url": "http://test.com",
                "auth_token": "test-token",
                "parameters": {},
            }
        }

        model_config = {"name": "test_unsupported", "model_type": "unsupported"}
        with self.assertRaises(NotImplementedError):
            ModelFactory.create_model(model_config)

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.validate_required_keys")
    def test_create_model_unsupported_backend(self, mock_validate, mock_load_model_config):
        """Test create_model with unsupported backend"""
        mock_load_model_config.return_value = {
            "test_model": {
                "model_type": "vllm",
                "url": "http://test.com",
                "auth_token": "test-token",
                "parameters": {},
            }
        }

        model_config = {"name": "test_model", "model_type": "vllm"}
        with self.assertRaises(NotImplementedError):
            ModelFactory.create_model(model_config, "unsupported_backend")

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.validate_required_keys")
    @patch("langchain_core.runnables.RunnableLambda")
    def test_get_model(self, mock_runnable, mock_validate, mock_load_model_config):
        """Test get_model method"""
        mock_load_model_config.return_value = {
            "test_model": {
                "model_type": "vllm",
                "url": "http://test.com",
                "auth_token": "test-token",
                "parameters": {},
            }
        }

        # Mock the model instance
        mock_model = MagicMock()
        with patch.object(ModelFactory, "create_model", return_value=mock_model) as mock_create:
            model_config = {"name": "test_model", "model_type": "vllm"}
            ModelFactory.get_model(model_config)

            # Verify create_model was called
            mock_create.assert_called_once_with(model_config, "default")

            # Verify RunnableLambda was called
            mock_runnable.assert_called()

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.validate_required_keys")
    def test_missing_required_keys(self, mock_validate, mock_load_model_config):
        """Test handling of missing required keys"""
        mock_load_model_config.return_value = {}

        # Set up validate_required_keys to raise ValueError
        mock_validate.side_effect = ValueError("Missing required key")

        with self.assertRaises(ValueError):
            ModelFactory.create_model({"name": "test_model"})

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.constants.BACKEND", "langgraph")
    def test_agent_node_model_initialization(self, mock_load_model_config):
        """Test that AgentNode uses the correct model type based on backend"""
        from grasp.core.graph.nodes.agent_node import AgentNode

        # Mock model config loading
        mock_load_model_config.return_value = {
            "test_model": {
                "model_type": "vllm",
                "url": "http://test.com",
                "auth_token": "test-token",
                "parameters": {},
            }
        }

        # Mock create_model to verify it gets called with the right parameters
        with patch.object(ModelFactory, "create_model", return_value=MagicMock()) as mock_create:
            # Create an agent node with minimal config
            node_config = {"model": {"name": "test_model"}, "prompt": "test prompt"}

            # Patch any other required methods that might be called during initialization
            with patch("grasp.utils.utils.get_graph_factory", return_value=MagicMock()):
                with patch("grasp.utils.utils.get_func_from_str", return_value=MagicMock()):
                    with patch("grasp.utils.utils.get_graph_properties", return_value={}):
                        AgentNode("test_node", node_config)

                        # Verify create_model was called with langgraph backend
                        mock_create.assert_called_with({"name": "test_model"}, "langgraph")


if __name__ == "__main__":
    unittest.main()
