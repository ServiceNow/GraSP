"""Local file system data handler implementation.

This module provides functionality for reading from and writing to local files
in various formats including JSON, JSONL, and Parquet.
"""

import json
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from grasp.core.dataset.data_handler_base import DataHandler
from grasp.core.dataset.dataset_config import DataSourceConfig, OutputConfig
from grasp.logger.logger_config import logger


class FileHandler(DataHandler):
    """Handler for interacting with local files.

    This class provides methods for reading from and writing to local files
    in various formats including JSON, JSONL, and Parquet.

    Args:
        source_config (Optional[DataSourceConfig]): Configuration for reading from files.
        output_config (Optional[OutputConfig]): Configuration for writing to files.

    Attributes:
        source_config (DataSourceConfig): Configuration for source files.
        output_config (OutputConfig): Configuration for output files.
    """

    def __init__(
        self,
        source_config: Optional[DataSourceConfig],
        output_config: Optional[OutputConfig] = None,
    ):
        self.source_config: Optional[DataSourceConfig] = source_config
        self.output_config: Optional[OutputConfig] = output_config

    def read(self, path: Union[str, PathLike[str], None] = None) -> list[dict[str, Any]]:
        """Read data from a local file.

        Supports reading from .parquet, .jsonl, and .json files.

        Args:
            path (Optional[str]): Path to the file. If None, uses path from source_config.

        Returns:
            list[dict[str, Any]]: List of records read from the file.

        Raises:
            ValueError: If file path is not provided or format is unsupported.
            Exception: If reading operation fails.
        """
        try:
            if path is None:
                if not self.source_config or not self.source_config.file_path:
                    raise ValueError("File path not provided")
                file_path = Path(self.source_config.file_path)
            else:
                file_path = Path(path)

            if file_path.suffix == ".parquet":
                return cast(
                    list[dict[str, Any]], pd.read_parquet(file_path).to_dict(orient="records")
                )
            elif file_path.suffix == ".csv":
                df = pd.read_csv(
                    file_path,
                    encoding=self.source_config.encoding if self.source_config else "utf-8",
                )
                return cast(list[dict[str, Any]], df.to_dict(orient="records"))
            elif file_path.suffix == ".jsonl":
                enc = self.source_config.encoding if self.source_config else "utf-8"
                with open(file_path, "r", encoding=enc) as f:
                    return cast(list[dict[str, Any]], [json.loads(line) for line in f])
            elif file_path.suffix == ".json":
                enc = self.source_config.encoding if self.source_config else "utf-8"
                with open(file_path, "r", encoding=enc) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return cast(list[dict[str, Any]], data)
                    raise ValueError("Expected a list of records in JSON file")
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        except Exception as e:
            logger.error(f"Failed to read file {path}: {str(e)}")
            raise

    def write(self, data: list[dict[str, Any]], path: str) -> None:
        """Write data to a local file.

        Supports writing to .parquet, .jsonl, and .json files.
        Creates parent directories if they don't exist.

        Args:
            data (list[dict[str, Any]]): Data to write.
            path (str): Path where the file should be written.

        Raises:
            Exception: If writing operation fails.
        """

        class JSONEncoder(json.JSONEncoder):
            """Custom JSON encoder for handling special data types."""

            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Writing {len(data)} records to local file: {output_path}")

            if output_path.suffix == ".parquet":
                df = pd.DataFrame(data)
                df.to_parquet(output_path)
            elif output_path.suffix == ".jsonl":
                enc = self.output_config.encoding if self.output_config else "utf-8"
                with open(output_path, "a", encoding=enc) as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False, cls=JSONEncoder) + "\n")
            else:
                enc = self.output_config.encoding if self.output_config else "utf-8"
                with open(output_path, "w", encoding=enc) as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, cls=JSONEncoder)

            logger.info(f"Successfully wrote data to {output_path}")

        except Exception as e:
            logger.error(f"Failed to write to file {path}: {str(e)}")
            raise

    def get_files(self) -> list[str]:
        """Get list of files matching configured pattern.

        Returns list of all files in the specified directory that match
        the configured pattern and extensions (.parquet, .jsonl, .json).

        Returns:
            list[str]: List of matching file paths.

        Raises:
            ValueError: If source directory is not configured.
        """
        if not self.source_config or not self.source_config.file_path:
            raise ValueError("Source directory not configured")

        source_dir = Path(self.source_config.file_path).parent
        # Use shard.regex from DataSourceConfig if provided; otherwise default to match all
        pattern = "*"
        if self.source_config and self.source_config.shard and self.source_config.shard.regex:
            pattern = self.source_config.shard.regex
        extensions = [".parquet", ".jsonl", ".json"]

        matching_files: list[Path] = []
        for ext in extensions:
            matching_files.extend(source_dir.glob(f"{pattern}*{ext}"))

        return [str(f) for f in matching_files]
