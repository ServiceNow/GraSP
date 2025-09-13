import json
import os
import shutil
import tempfile
import time
from typing import Any, Optional, Union

from grasp.core.dataset.dataset_config import DataSourceType, OutputType
from grasp.core.dataset.file_handler import FileHandler
from grasp.core.dataset.huggingface_handler import HuggingFaceHandler
from grasp.logger.logger_config import logger
from grasp.tools.config import Config
from grasp.tools.registry import ToolRegistry
from grasp.utils import constants


class ToolExecutor:
    def __init__(self, config: Union[str, dict, Config]):
        """Initialize the ToolExecutor with a configuration."""
        ToolRegistry.auto_discover()

        if isinstance(config, str):
            self.config = Config.from_file(config)
        elif isinstance(config, dict):
            self.config = Config.from_dict(config)
        else:
            self.config = config

        self.source_config = (
            self.config.data_config.get_source_config() if self.config.data_config else None
        )
        self.sink_config = (
            self.config.data_config.get_sink_config() if self.config.data_config else None
        )

        self.temp_dir = tempfile.TemporaryDirectory()
        self.current_path: Optional[str] = None
        self.final_output_path: Optional[str] = None
        self.available_tool_names = ToolRegistry.available_tools()

    def execute(self) -> str:
        self._load_input()
        self._run_tool_chain()
        self._handle_output_sink()
        if self.final_output_path is None:
            raise ValueError("No final output path")
        return self.final_output_path

    def _load_input(self) -> None:
        if not self.source_config:
            raise ValueError("No data source configuration provided")

        logger.info(f"Loading data from source: {self.source_config.type}")

        if self.source_config.type == DataSourceType.HUGGINGFACE:
            data = HuggingFaceHandler(self.source_config).read()
            # Convert potential iterator to a concrete list for typing correctness
            records: list[dict[str, Any]] = data if isinstance(data, list) else list(data)
            self.current_path = os.path.join(self.temp_dir.name, "input.jsonl")
            file_handler = FileHandler(self.source_config, self.sink_config)
            file_handler.write(records, self.current_path)

        elif self.source_config.type == DataSourceType.DISK_FILE:
            path = self.source_config.file_path
            if not path:
                raise ValueError("Input file_path must be set for DISK_FILE source")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Input file not found: {path}")
            self.current_path = path

        else:
            raise ValueError(f"Unsupported source type: {self.source_config.type}")

    def _run_tool_chain(self) -> None:
        if not self.config.tools:
            logger.warning("No tools defined in configuration")
            return
        if self.current_path is None:
            logger.warning("No input path provided")
            return
        tool_names = list(self.config.tools.keys())

        for step_index, (tool_name, tool_config) in enumerate(self.config.tools.items()):
            tool_class = ToolRegistry.get_tool(tool_name)
            if not tool_class:
                logger.error(
                    f"Tool '{tool_name}' not found. Available: {self.available_tool_names}"
                )
                continue

            is_last_tool = tool_name == tool_names[-1]
            output_path = self._resolve_output_path(tool_name, step_index, is_last_tool)

            logger.info(f"\nExecuting tool '{tool_name}' (step {step_index + 1})")
            logger.info(f"Input: {self.current_path} → Output: {output_path}")

            try:
                tool = tool_class(tool_config.config)
                start_time = time.time()
                tool.process(input_path=self.current_path, output_path=output_path)
                logger.info(f"'{tool_name}' complete in {time.time() - start_time:.2f}s")
                self.current_path = output_path
                self.final_output_path = output_path
            except Exception as e:
                logger.error(f"Tool '{tool_name}' failed: {e}")
                raise

    def _resolve_output_path(self, tool_name: str, step_index: int, is_last_tool: bool) -> str:
        if is_last_tool:
            if self.sink_config and self.sink_config.type != OutputType.HUGGINGFACE:
                if self.sink_config.file_path is None:
                    raise ValueError("Output file path is required for non-Hugging Face sink")
                return self.sink_config.file_path
            else:
                output_dir = os.path.join(constants.ROOT_DIR, "tools", "tool_runs")
                os.makedirs(output_dir, exist_ok=True)
                return os.path.join(
                    output_dir,
                    f"final_output_{tool_name}_{time.strftime('%Y%m%d-%H%M%S')}.json",
                )
        else:
            return os.path.join(self.temp_dir.name, f"output_{step_index}.jsonl")

    def _handle_output_sink(self) -> None:
        if not self.sink_config:
            return

        if self.sink_config.type == OutputType.HUGGINGFACE:
            self._upload_to_huggingface()
        elif self.sink_config.type != OutputType.HUGGINGFACE:
            self._copy_to_sink()

    def _upload_to_huggingface(self) -> None:
        try:
            if self.sink_config is None:
                raise ValueError("Output sink_config is missing")
            if self.sink_config.repo_id is None:
                raise ValueError("Output repo_id is required in sink_config for Hugging Face sink")
            if self.final_output_path is None:
                raise ValueError("Output file path to be loaded for Hugging Face sink is missing")
            logger.info(f"Uploading to Hugging Face: {self.sink_config.repo_id}")
            with open(self.final_output_path, "r", encoding="utf-8") as f:
                data = (
                    json.load(f)
                    if self.final_output_path.endswith(".json")
                    else [json.loads(line) for line in f if line.strip()]
                )
            HuggingFaceHandler(self.source_config, self.sink_config).write(data)
            logger.info("Upload complete")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise

    def _copy_to_sink(self) -> None:
        if self.sink_config is None:
            raise ValueError("Output sink_config is missing")
        sink_path_opt = self.sink_config.file_path
        if sink_path_opt is None:
            raise ValueError("Output file path is required for non-Hugging Face sink")
        if self.final_output_path is None:
            raise ValueError("final_output_path is not set before copying to sink")

        sink_path: str = sink_path_opt
        final_path: str = self.final_output_path

        if os.path.abspath(sink_path) != os.path.abspath(final_path):
            os.makedirs(os.path.dirname(sink_path), exist_ok=True)
            shutil.copyfile(final_path, sink_path)
            logger.info(f"Output copied to sink: {sink_path}")
            self.final_output_path = sink_path
        else:
            logger.info(f"Output already at sink location: {sink_path}")


def run_tool(
    tool_name: str,
    input_path: str,
    config: Optional[Any] = None,
    output_path: Optional[str] = None,
) -> str:
    """Convenience method to run a single tool."""
    data_config: dict[str, Any] = {"source": {"type": "disk", "file_path": input_path}}
    config_dict: dict[str, Any] = {
        "data_config": data_config,
        "tools": {tool_name: {"config": config or {}}},
    }

    if output_path:
        data_config["sink"] = {"type": "jsonl", "file_path": output_path}

    executor = ToolExecutor(config_dict)
    return executor.execute()
