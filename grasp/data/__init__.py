"""
Data handling API for GraSP workflows with full framework integration.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

try:
    from grasp.core.dataset.dataset_config import DataSourceConfig  # noqa: F401
    from grasp.core.dataset.dataset_config import DataSourceType  # noqa: F401
    from grasp.core.dataset.dataset_config import OutputConfig  # noqa: F401
    from grasp.core.dataset.dataset_config import OutputType  # noqa: F401
    from grasp.core.dataset.dataset_config import ShardConfig  # noqa: F401
    from grasp.core.dataset.dataset_config import TransformConfig  # noqa: F401
    from grasp.core.dataset.file_handler import FileHandler  # noqa: F401
    from grasp.core.dataset.huggingface_handler import HuggingFaceHandler  # noqa: F401

    CORE_DATA_AVAILABLE = True
except ImportError:
    CORE_DATA_AVAILABLE = False


class DataSource:
    """Data source abstraction with full framework integration."""

    def __init__(self, config: dict[str, Any]):
        self._config = config

    def to_config(self) -> dict[str, Any]:
        """Return the configuration dict."""
        return self._config

    @classmethod
    def memory(cls, data: list[dict[str, Any]]) -> "DataSource":
        """Create in-memory data source from list."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        with temp_file as f:
            json.dump(data, f, indent=2)

        return cls(
            {
                "type": "disk",
                "file_path": temp_file.name,
                "file_format": "json",
                "_temp_file": True,
            }
        )

    @classmethod
    def disk(cls, path: Union[str, Path], **kwargs) -> "DataSource":
        """Create file-based data source with framework support."""
        path = Path(path)
        file_format = path.suffix.lstrip(".") or "json"

        config = {
            "type": "disk",
            "file_path": str(path),
            "file_format": file_format,
            "encoding": kwargs.get("encoding", "utf-8"),
        }

        # Add transformation support
        if "transformations" in kwargs:
            config["transformations"] = kwargs["transformations"]

        # Add shard support
        if "shard" in kwargs:
            config["shard"] = kwargs["shard"]

        return cls(config)

    @classmethod
    def huggingface(cls, repo_id: str, **kwargs) -> "DataSource":
        """Create HuggingFace data source with full framework support."""
        config = {
            "type": "hf",
            "repo_id": repo_id,
            "config_name": kwargs.get("config_name"),
            "split": kwargs.get("split", "train"),
            "token": kwargs.get("token"),
            "streaming": kwargs.get("streaming", False),
        }

        # Add shard support for HuggingFace
        if "shard" in kwargs:
            config["shard"] = kwargs["shard"]

        # Add transformation support
        if "transformations" in kwargs:
            config["transformations"] = kwargs["transformations"]

        return cls(config)

    def add_transformation(
        self, transform: str, params: Optional[dict[str, Any]] = None
    ) -> "DataSource":
        """Add data transformation."""
        if "transformations" not in self._config:
            self._config["transformations"] = []

        transform_config: dict[str, Any] = {"transform": transform}
        if params:
            transform_config["params"] = params

        self._config["transformations"].append(transform_config)
        return self

    def add_shard(self, regex: str = "*", index: Optional[list[int]] = None) -> "DataSource":
        """Add shard configuration."""
        shard_config: dict[str, Any] = {"regex": regex}
        if index:
            shard_config["index"] = index

        self._config["shard"] = shard_config
        return self


class DataSink:
    """Data sink abstraction with full framework integration."""

    def __init__(self, config: dict[str, Any]):
        self._config = config

    def to_config(self) -> dict[str, Any]:
        """Return the configuration dict."""
        return self._config

    @classmethod
    def disk(cls, path: Union[str, Path], **kwargs) -> "DataSink":
        """Create file-based data sink with framework support."""
        path = Path(path)
        file_format = "json" if path.suffix == ".json" else "jsonl"

        # Ensure directory exists
        os.makedirs(path.parent, exist_ok=True)

        config = {
            "type": file_format,
            "file_path": str(path),
            "encoding": kwargs.get("encoding", "utf-8"),
        }

        # Add chunk size for large datasets
        if "chunk_size" in kwargs:
            config["chunk_size"] = kwargs["chunk_size"]

        return cls(config)

    @classmethod
    def huggingface(cls, repo_id: str, **kwargs) -> "DataSink":
        """Create HuggingFace data sink with full framework support."""
        config = {
            "type": "hf",
            "repo_id": repo_id,
            "config_name": kwargs.get("config_name"),
            "split": kwargs.get("split", "train"),
            "token": kwargs.get("token"),
            "private": kwargs.get("private", True),
            "chunk_size": kwargs.get("chunk_size", 1000),
        }

        # Add filename support
        if "filename" in kwargs:
            config["filename"] = kwargs["filename"]

        return cls(config)


class DataSourceFactory:
    """Factory for creating data source configurations with framework integration."""

    @staticmethod
    def from_file(path: Union[str, Path], **kwargs) -> dict[str, Any]:
        """Create file data source configuration."""
        return DataSource.disk(path, **kwargs).to_config()

    @staticmethod
    def from_huggingface(repo_id: str, **kwargs) -> dict[str, Any]:
        """Create HuggingFace data source configuration."""
        return DataSource.huggingface(repo_id, **kwargs).to_config()

    @staticmethod
    def from_list(data: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """Create in-memory data source configuration."""
        return DataSource.memory(data).to_config()

    @staticmethod
    def from_iterable_dataset(dataset, **kwargs) -> dict[str, Any]:
        """Create configuration for HuggingFace IterableDataset."""
        if CORE_DATA_AVAILABLE:
            # Use framework's dataset handling
            return {
                "type": "hf",
                "streaming": True,
                "dataset_object": dataset,
                **kwargs,
            }
        else:
            raise NotImplementedError("IterableDataset support requires core framework")


class DataSinkFactory:
    """Factory for creating data sink configurations with framework integration."""

    @staticmethod
    def to_file(path: Union[str, Path], **kwargs) -> dict[str, Any]:
        """Create file data sink configuration."""
        return DataSink.disk(path, **kwargs).to_config()

    @staticmethod
    def to_huggingface(repo_id: str, **kwargs) -> dict[str, Any]:
        """Create HuggingFace data sink configuration."""
        return DataSink.huggingface(repo_id, **kwargs).to_config()


# Enhanced convenience functions with framework features
def from_file(
    path: Union[str, Path],
    transformations: Optional[list[dict[str, Any]]] = None,
    shard: Optional[dict[str, Any]] = None,
    **kwargs,
) -> dict[str, Any]:
    """Create a data source from a file with optional transformations and sharding."""
    source = DataSource.disk(path, **kwargs)

    if transformations:
        for transform in transformations:
            source.add_transformation(transform["transform"], transform.get("params"))

    if shard:
        source.add_shard(shard.get("regex", "*"), shard.get("index"))

    return source.to_config()


def from_huggingface(
    repo_id: str,
    transformations: Optional[list[dict[str, Any]]] = None,
    shard: Optional[dict[str, Any]] = None,
    **kwargs,
) -> dict[str, Any]:
    """Create a data source from HuggingFace dataset with optional transformations and sharding."""
    source = DataSource.huggingface(repo_id, **kwargs)

    if transformations:
        for transform in transformations:
            source.add_transformation(transform["transform"], transform.get("params"))

    if shard:
        source.add_shard(shard.get("regex", "*"), shard.get("index"))

    return source.to_config()


def from_list(data: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
    """Create a data source from a Python list."""
    return DataSourceFactory.from_list(data, **kwargs)


def to_file(path: Union[str, Path], **kwargs) -> dict[str, Any]:
    """Create a data sink to a file."""
    return DataSinkFactory.to_file(path, **kwargs)


def to_huggingface(repo_id: str, **kwargs) -> dict[str, Any]:
    """Create a data sink to HuggingFace dataset."""
    return DataSinkFactory.to_huggingface(repo_id, **kwargs)


__all__ = [
    "DataSource",
    "DataSink",
    "DataSourceFactory",
    "DataSinkFactory",
    "from_file",
    "from_huggingface",
    "from_list",
    "to_file",
    "to_huggingface",
]
