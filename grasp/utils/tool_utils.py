import importlib
from abc import ABC, abstractmethod
from inspect import getmembers, isclass
from typing import List

from langchain_core.tools import BaseTool

from grasp.logger.logger_config import logger
from grasp.utils import utils


class LangChainToolsWrapper(ABC):
    """
    Inherit this class and implement get_tools()
    Which returns tools list to be used directly in the react agent
    """

    @abstractmethod
    def get_tools(self) -> list:
        pass


def load_tools(tool_paths: List[str]) -> List[BaseTool]:
    """
    Load tools from a variety of sources specified in tool_paths.

    This method processes each path and determines whether it's:
    - An individual tool function (has dots and the last component is not a class)
    - A module (to load all tools from)
    - A class (to load all tool methods from)

    Args:
        tool_paths: List of paths that could be individual tools,
                   modules, classes, or files

    Returns:
        List of loaded tools from all sources
    """
    tools: List[BaseTool] = []

    for path in tool_paths:
        # First, check if it's an individual tool function using dot notation
        try:
            valid = False
            # Check for standard import path with dots
            if "." in path:
                # Try to load as an individual function first
                try:
                    tool_func = utils.get_func_from_str(path)
                    # If it's a tool, add it
                    if isinstance(tool_func, BaseTool):
                        tools.append(tool_func)
                        valid = True
                        continue
                except (ImportError, AttributeError):
                    # If it fails, it might be a module or class instead
                    pass

                # Try as a class
                module_path = ".".join(path.split(".")[:-1])
                obj_name = path.split(".")[-1]

                try:
                    # Import the module
                    module = importlib.import_module(module_path)

                    # Check if the object is a class
                    if hasattr(module, obj_name):
                        obj = getattr(module, obj_name)
                        if isclass(obj):
                            # It's a class, get all tool methods
                            tools.extend(_extract_tools_from_class(obj))
                            valid = True
                            continue
                        elif isinstance(tool_func, type) and issubclass(
                            tool_func, LangChainToolsWrapper
                        ):
                            tools.extend(tool_func().get_tools())
                            valid = True
                            continue
                except (ImportError, AttributeError):
                    pass

                # Try as a module
                try:
                    module = importlib.import_module(path)
                    tools.extend(_extract_tools_from_module(module))
                    valid = True
                    continue
                except (ImportError, AttributeError):
                    pass
            else:
                logger.warn(f"Tool path '{path}' is not a valid import path. Skipping...")
                continue

            if not valid:
                logger.error(f"Failed to load tool from path '{path}'")
                raise ValueError(f"Failed to load tool from path '{path}'")

        except Exception as e:
            logger.error(f"Failed to load tool from path '{path}': {str(e)}")
            raise ValueError(f"Failed to load tool from path '{path}': {str(e)}")
    # if len(tools) == 0:
    #     logger.warn("No tools found in tool paths. Refer to tools defined in graph_config")
    return tools


def _extract_tools_from_module(module) -> List[BaseTool]:
    """
    Extract all tool functions from a module.

    Args:
        module: The module to extract tools from

    Returns:
        List of tool functions found in the module
    """
    tools: List[BaseTool] = []

    # Find all functions in the module that have been decorated with @tool
    for _, obj in getmembers(module):
        if isinstance(obj, BaseTool):
            tools.append(obj)

    return tools


def _extract_tools_from_class(cls) -> List[BaseTool]:
    """
    Extract all tool methods from a class.

    Args:
        cls: The class to extract tool methods from

    Returns:
        List of tool methods found in the class
    """
    tools: List[BaseTool] = []

    # Find all methods in the class that have been decorated with @tool
    for _, obj in getmembers(cls):
        if isinstance(obj, BaseTool):
            tools.append(obj)

    return tools
