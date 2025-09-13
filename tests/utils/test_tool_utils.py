import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.tools import BaseTool, Tool

sys.path.append(str(Path(__file__).parent.parent.parent))

from grasp.utils import utils
from grasp.utils.tool_utils import (
    _extract_tools_from_class,
    _extract_tools_from_module,
    load_tools,
)


class TestToolUtils(unittest.TestCase):
    def test_extract_tools_from_module(self):
        """Test extracting tools from a module"""
        # Create a mock module with tool instances
        mock_module = MagicMock()
        mock_tool1 = Tool(name="tool1", func=lambda x: x, description="Tool 1")
        mock_tool2 = Tool(name="tool2", func=lambda x: x, description="Tool 2")

        # Set up mock module with members including tools and non-tools
        mock_module_members = [
            ("tool1", mock_tool1),
            ("tool2", mock_tool2),
            ("not_a_tool", "string_value"),
            ("another_non_tool", lambda: None),
        ]

        with patch("grasp.utils.tool_utils.getmembers", return_value=mock_module_members):
            tools = _extract_tools_from_module(mock_module)

        self.assertEqual(len(tools), 2)
        self.assertIn(mock_tool1, tools)
        self.assertIn(mock_tool2, tools)

    def test_extract_tools_from_class(self):
        """Test extracting tools from a class"""
        # Create a mock class with tool instances
        mock_class = MagicMock()
        mock_tool1 = Tool(name="class_tool1", func=lambda x: x, description="Class Tool 1")
        mock_tool2 = Tool(name="class_tool2", func=lambda x: x, description="Class Tool 2")

        # Set up mock class with members including tools and non-tools
        mock_class_members = [
            ("tool1", mock_tool1),
            ("tool2", mock_tool2),
            ("not_a_tool", "string_value"),
            ("method", lambda self: None),
        ]

        with patch("grasp.utils.tool_utils.getmembers", return_value=mock_class_members):
            tools = _extract_tools_from_class(mock_class)

        self.assertEqual(len(tools), 2)
        self.assertIn(mock_tool1, tools)
        self.assertIn(mock_tool2, tools)

    def test_load_tools_from_individual_function(self):
        """Test loading an individual tool function"""
        # Create a mock tool
        mock_tool = MagicMock(spec=BaseTool)

        # Use context managers for patching to ensure proper control flow
        with patch.object(utils, "get_func_from_str", return_value=mock_tool) as mock_get_func:
            # The isinstance check in the code needs to pass for our mock_tool
            with patch("grasp.utils.tool_utils.isinstance", return_value=True):
                tools = load_tools(["module.submodule.tool_func"])

                # Assert that the get_func_from_str was called with the right path
                mock_get_func.assert_called_once_with("module.submodule.tool_func")
                self.assertEqual(len(tools), 1)
                self.assertEqual(tools[0], mock_tool)

    @patch("grasp.utils.tool_utils.logger")
    def test_load_tools_with_invalid_path(self, mock_logger):
        """Test loading tools with an invalid path format"""
        tools = load_tools(["invalid_path_no_dots"])

        # Should log a warning
        mock_logger.warn.assert_called_once()
        self.assertEqual(len(tools), 0)

    @patch("grasp.utils.tool_utils.utils.get_func_from_str")
    @patch("grasp.utils.tool_utils.importlib.import_module")
    @patch("grasp.utils.tool_utils.logger")
    def test_load_tools_with_import_error(self, mock_logger, mock_import_module, mock_get_func):
        """Test handling of ImportError when loading tools"""
        mock_get_func.side_effect = ImportError("Module not found")
        mock_import_module.side_effect = ImportError("Module not found")

        with self.assertRaises(ValueError):
            load_tools(["module.not.found"])

        # Should log a warning
        mock_logger.error.assert_called()


if __name__ == "__main__":
    unittest.main()
