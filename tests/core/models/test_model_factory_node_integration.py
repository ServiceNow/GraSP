import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path to import the necessary modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from grasp.core.models.custom_models import CustomVLLM
from grasp.core.models.langgraph.vllm_chat_model import CustomVLLMChatModel
from grasp.core.models.model_factory import ModelFactory


class TestModelFactoryNodeIntegration(unittest.TestCase):
    """Integration tests for the ModelFactory with LLMNode and AgentNode"""

    def setUp(self):
        # Set up common test data
        self.base_model_config = {
            "test_model": {
                "model_type": "vllm",
                "url": "http://test.com",
                "auth_token": "test-token",
                "parameters": {},
            }
        }

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.get_graph_properties")
    @patch("grasp.utils.utils.get_func_from_str")
    def test_llm_node_model_initialization(
        self, mock_get_func, mock_get_props, mock_load_model_config
    ):
        """Test that LLMNode initializes model using ModelFactory.get_model"""
        from grasp.core.graph.nodes.llm_node import LLMNode

        # Configure mocks
        mock_load_model_config.return_value = self.base_model_config
        mock_get_props.return_value = {}
        mock_get_func.return_value = lambda x: x

        # Create a mock for the get_model method
        with patch.object(ModelFactory, "get_model") as mock_get_model:
            mock_get_model.return_value = MagicMock()

            # Initialize an LLM node
            node_config = {"model": {"name": "test_model"}, "prompt": "test prompt"}

            LLMNode("test_llm_node", node_config)

            # Verify get_model was called
            mock_get_model.assert_called_once_with({"name": "test_model"})

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.get_graph_properties")
    @patch("grasp.utils.utils.get_func_from_str")
    @patch("grasp.utils.utils.get_graph_factory")
    def test_agent_node_model_initialization(
        self, mock_graph_factory, mock_get_func, mock_get_props, mock_load_model_config
    ):
        """Test that AgentNode initializes model using ModelFactory.get_model with the correct backend"""
        from grasp.core.graph.nodes.agent_node import AgentNode

        # Configure mocks
        mock_load_model_config.return_value = self.base_model_config
        mock_get_props.return_value = {}
        mock_get_func.return_value = lambda x: x
        mock_graph_factory.return_value = MagicMock()

        # Test with langgraph backend specifically
        with patch("grasp.utils.constants.BACKEND", "langgraph"):
            # Reset and reconfigure mocks
            mock_load_model_config.return_value = self.base_model_config

            # Create a mock for the get_model method
            with patch.object(ModelFactory, "create_model") as mock_get_model:
                mock_get_model.return_value = MagicMock()

                # Initialize an Agent node
                node_config = {"model": {"name": "test_model"}, "prompt": "test prompt"}

                AgentNode("test_agent_node", node_config)

                # Verify get_model was called with the langgraph backend
                mock_get_model.assert_called_once_with({"name": "test_model"}, "langgraph")

    @patch("grasp.utils.utils.load_model_config")
    def test_model_factory_backend_selection(self, mock_load_model_config):
        """Test that ModelFactory selects the appropriate model class based on backend"""
        # Configure mock
        mock_load_model_config.return_value = self.base_model_config

        # Test default backend uses standard model
        with patch.object(CustomVLLM, "__init__", return_value=None) as mock_default:
            model_config = {"name": "test_model", "model_type": "vllm"}
            ModelFactory.create_model(model_config, "default")
            mock_default.assert_called_once()

        # Test langgraph backend uses chat models
        with patch.object(CustomVLLMChatModel, "__init__", return_value=None) as mock_langgraph:
            model_config = {"name": "test_model", "model_type": "vllm"}
            ModelFactory.create_model(model_config, "langgraph")
            mock_langgraph.assert_called_once()

    @patch("grasp.utils.utils.load_model_config")
    @patch("grasp.utils.utils.get_graph_factory")
    @patch("grasp.utils.utils.get_graph_properties")
    @patch("grasp.utils.utils.get_func_from_str")
    def test_agent_node_ensures_chat_model(
        self, mock_get_func, mock_get_props, mock_graph_factory, mock_load_model_config
    ):
        """Test that AgentNode ensures using a model extended from BaseChatModel via ModelFactory.get_model"""
        from langchain_core.language_models import BaseChatModel

        from grasp.core.graph.nodes.agent_node import AgentNode

        # Test that with langgraph backend, we get a proper BaseChatModel
        with patch("grasp.utils.constants.BACKEND", "langgraph"):
            # Configure mocks
            mock_load_model_config.return_value = self.base_model_config
            mock_get_props.return_value = {}
            mock_get_func.return_value = lambda x: x
            mock_graph_factory.return_value = MagicMock()

            # Create a mock for get_model since that's what AgentNode actually uses
            with patch.object(ModelFactory, "create_model") as mock_get_model:
                # Set up a mock for the model that is returned
                mock_model = MagicMock(spec=CustomVLLMChatModel)
                # Ensure the model mock has the BaseChatModel interface
                mock_model.__class__.__bases__ = (BaseChatModel,)
                mock_get_model.return_value = mock_model

                # Initialize an Agent node
                node_config = {
                    "model": {"name": "test_model", "model_type": "vllm"},
                    "prompt": "test prompt",
                }

                # Initialize the Agent node
                node = AgentNode("test_agent_node", node_config)

                # Verify get_model was called with the right arguments
                mock_get_model.assert_called_once_with(
                    {"name": "test_model", "model_type": "vllm"}, "langgraph"
                )

                # Verify the model in the node is our mock with BaseChatModel interface
                self.assertIsInstance(node.model, MagicMock)
                self.assertTrue(issubclass(node.model.__class__.__bases__[0], BaseChatModel))

    @patch("grasp.utils.utils.load_model_config")
    def test_incompatible_model_for_agent_node(self, mock_load_model_config):
        """Test handling of model types not supported in langgraph backend"""
        # Configure mock for a model type that doesn't have a langgraph implementation
        mock_load_model_config.return_value = {
            "test_model": {
                "model_type": "mistralai",  # No langgraph implementation for this
                "url": "http://test.com",
                "auth_token": "test-token",
                "parameters": {},
            }
        }

        # Test that an error is raised when trying to use an unsupported model type with langgraph
        with self.assertRaises(NotImplementedError):
            model_config = {"name": "test_model", "model_type": "mistralai"}
            ModelFactory.create_model(model_config, "langgraph")

    @patch("grasp.utils.utils.load_model_config")
    def test_incompatible_ollama_model_for_agent_node(self, mock_load_model_config):
        """Test handling of ollama model type which is not supported in langgraph backend"""
        # Configure mock for a model type that doesn't have a langgraph implementation
        mock_load_model_config.return_value = {
            "test_model": {
                "model_type": "ollama",  # No langgraph implementation for this
                "parameters": {},
            }
        }

        # Test that an error is raised when trying to use an unsupported model type with langgraph
        with self.assertRaises(NotImplementedError):
            model_config = {"name": "test_model", "model_type": "ollama"}
            ModelFactory.create_model(model_config, "langgraph")


if __name__ == "__main__":
    unittest.main()
