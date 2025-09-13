import os
from pathlib import Path
from typing import Any, Union, cast

import yaml

try:
    from grasp.core.dataset.dataset_config import DataSourceConfig, OutputConfig  # noqa: F401
    from grasp.core.graph.graph_config import GraphConfig  # noqa: F401
    from grasp.utils import utils

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


class ConfigLoader:
    """Loads existing GraSP YAML configurations and converts them to library objects."""

    def load(self, config_path: Union[str, Path, dict[str, Any]]) -> dict[str, Any]:
        """Load configuration from file or dictionary."""
        if isinstance(config_path, dict):
            return config_path

        config_path = Path(config_path)
        if not config_path.exists():
            if UTILS_AVAILABLE:
                task_config_path = utils.get_file_in_task_dir(config_path.stem, "graph_config.yaml")
                if os.path.exists(task_config_path):
                    config_path = Path(task_config_path)
                else:
                    raise FileNotFoundError(f"Configuration file not found: {config_path}")
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config = cast(dict[str, Any], yaml.safe_load(f))

        return config

    def load_and_create(self, config_path: Union[str, Path, dict[str, Any]]):
        """Load config and create appropriate Workflow or Graph object."""
        config = self.load(config_path)

        # Import here to avoid circular imports
        from ..workflow import Workflow

        workflow = Workflow()
        workflow._config = config

        workflow._supports_subgraphs = True
        workflow._supports_multimodal = True
        workflow._supports_resumable = True
        workflow._supports_quality = True
        workflow._supports_oasst = True

        if isinstance(config_path, (str, Path)):
            workflow.name = Path(config_path).parent.name
        else:
            workflow.name = config.get("task_name", "loaded_workflow")

        return workflow


__all__ = ["ConfigLoader"]
