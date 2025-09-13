"""HuggingFace dataset handler implementation.

This module provides functionality for reading from and writing to HuggingFace datasets,
including support for sharded datasets and dataset card management.
"""

import base64
import io
import os
from typing import Any, Iterator, Optional, Union, cast

import datasets  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
from datasets import Dataset, IterableDataset, concatenate_datasets
from datasets import config as ds_config
from datasets import load_dataset
from datasets.utils.metadata import MetadataConfigs  # type: ignore[import-untyped]
from huggingface_hub import (
    CommitOperationAdd,
    DatasetCard,
    DatasetCardData,
    HfApi,
    HfFileSystem,
)

from grasp.core.dataset.data_handler_base import DataHandler
from grasp.core.dataset.dataset_config import DataSourceConfig, OutputConfig
from grasp.logger.logger_config import logger

hf_token = os.getenv("GRASP_HF_TOKEN")


class HuggingFaceHandler(DataHandler):
    """Handler for interacting with HuggingFace datasets.

    This class provides methods for reading from and writing to HuggingFace datasets,
    with support for sharded datasets, streaming, and dataset card management.

    Args:
        source_config (Optional[DataSourceConfig]): Configuration for reading from HuggingFace.
        output_config (Optional[OutputConfig]): Configuration for writing to HuggingFace.

    Attributes:
        source_config (DataSourceConfig): Configuration for source dataset.
        output_config (OutputConfig): Configuration for output dataset.
        fs (HfFileSystem): HuggingFace filesystem interface.
    """

    def __init__(
        self,
        source_config: Optional[DataSourceConfig],
        output_config: Optional[OutputConfig] = None,
    ):
        self.source_config: Optional[DataSourceConfig] = source_config
        self.output_config: Optional[OutputConfig] = output_config
        token = source_config.token if source_config else None
        self.fs = HfFileSystem(token=token)

    def read(
        self, path: Optional[str] = None
    ) -> Union[list[dict[str, Any]], Iterator[dict[str, Any]]]:
        """Read data from HuggingFace dataset.

        Args:
            path (Optional[str]): Path to specific shard file if reading sharded data.

        Returns:
            Union[list[dict[str, Any]], Iterator[dict[str, Any]]]: Dataset records.

        Raises:
            ValueError: If required configuration is missing.
            RuntimeError: If reading operation fails.
        """
        try:
            if not self.source_config:
                raise ValueError("Source configuration is required to read from HuggingFace")
            sc = self.source_config
            if path and sc.shard:
                return self._read_shard(path)
            return self._read_dataset()
        except Exception as e:
            logger.error(f"Failed to read from HuggingFace: {str(e)}")
            raise RuntimeError(f"Failed to read from HuggingFace: {str(e)}") from e

    def write(self, data: list[dict[str, Any]], path: Optional[str] = None) -> None:
        """
        Write data to a HuggingFace dataset.

        Args:
            data (list[dict[str, Any]]): Data to write.
            path (str, optional): Not used for HuggingFace, but required by interface.

        Raises:
            ValueError: If output configuration is missing.
            RuntimeError: If writing operation fails.
        """
        if not self.output_config:
            raise ValueError("Output configuration required for writing to HuggingFace")

        try:
            self._create_repo()

            df = pd.DataFrame(data)

            media_columns = self._detect_media_columns(df)

            for col in media_columns["image_seq"] + media_columns["audio_seq"]:
                df[col] = df[col].apply(self._decode_base64_media)

            for col in media_columns["image_str"] + media_columns["audio_str"]:
                df[col] = df[col].apply(
                    lambda x: self._decode_base64_media(x)[0] if isinstance(x, str) else None
                )

            ds = Dataset.from_pandas(df)

            ds = self._cast_dataset_columns(ds, media_columns)

            ds.push_to_hub(
                repo_id=self.output_config.repo_id,
                config_name=self.output_config.config_name,
                split=self.output_config.split,
                private=self.output_config.private,
                token=self.output_config.token,
            )

            self._update_readme_config()
            logger.info(
                f"Successfully wrote output to HuggingFace dataset: "
                f"{self.output_config.repo_id}, {self.output_config.config_name}"
            )

        except Exception as e:
            logger.error(f"Failed to write to HuggingFace: {str(e)}")
            raise RuntimeError("Writing to HuggingFace failed") from e

    def _create_repo(self) -> None:
        if not self.output_config:
            raise ValueError("Output configuration is required to create HuggingFace repo")
        oc = self.output_config
        if not oc.repo_id:
            raise ValueError("Output configuration must include a non-empty repo_id")
        repo_id: str = str(oc.repo_id)
        private: bool = bool(oc.private)
        api = HfApi(token=hf_token)
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )

    def _detect_media_columns(self, df: pd.DataFrame) -> dict[str, list[str]]:
        media_cols: dict[str, list[str]] = {
            "image_str": [],
            "image_seq": [],
            "audio_str": [],
            "audio_seq": [],
        }

        for col in df.columns:
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None

            if isinstance(sample, list):
                if all(isinstance(x, str) and x.startswith("data:image/") for x in sample):
                    media_cols["image_seq"].append(col)
                elif all(isinstance(x, str) and x.startswith("data:audio/") for x in sample):
                    media_cols["audio_seq"].append(col)
            elif isinstance(sample, str):
                if sample.startswith("data:image/"):
                    media_cols["image_str"].append(col)
                elif sample.startswith("data:audio/"):
                    media_cols["audio_str"].append(col)

        return media_cols

    def _decode_base64_media(self, val: Any) -> list[Optional[dict[str, bytes]]]:
        results: list[Optional[dict[str, bytes]]] = []
        if isinstance(val, str) and val.startswith("data:"):
            val = [val]
        if isinstance(val, list):
            for item in val:
                try:
                    if isinstance(item, str) and item.startswith("data:"):
                        _, encoded = item.split(",", 1)
                        results.append({"bytes": base64.b64decode(encoded)})
                    else:
                        results.append(None)
                except Exception:
                    results.append(None)
        return results

    def _cast_dataset_columns(self, ds: Dataset, media_cols: dict) -> Dataset:
        for col in media_cols["image_str"]:
            ds = ds.cast_column(col, datasets.Image())
        for col in media_cols["image_seq"]:
            ds = ds.cast_column(col, datasets.Sequence(datasets.Image()))

        for col in media_cols["audio_str"]:
            ds = ds.cast_column(col, datasets.Audio())
        for col in media_cols["audio_seq"]:
            ds = ds.cast_column(col, datasets.Sequence(datasets.Audio()))

        return ds

    def get_files(self) -> list[str]:
        """Get list of dataset files from HuggingFace.

        If sharding is enabled, returns list of shard files.
        Otherwise, returns list of all dataset files.

        Returns:
            list[str]: List of file paths in the dataset.

        Raises:
            RuntimeError: If operation fails.
        """
        try:
            if not self.source_config:
                raise ValueError("Source configuration is required to list files from HuggingFace")
            sc = self.source_config
            if sc.shard:
                pattern = f"datasets/{sc.repo_id}/{sc.config_name}/{sc.split}"
                pattern = f"{pattern}{sc.shard.regex}"
            else:
                pattern = f"datasets/{sc.repo_id}/{sc.config_name}/*"

            all_files = self.fs.glob(pattern)
            logger.info(f"Found {len(all_files)} total files")

            if sc.shard and sc.shard.index:
                filtered_files = [f for i, f in enumerate(all_files) if i in sc.shard.index]
                logger.info(f"Filtered to {len(filtered_files)} files based on index")
                return filtered_files

            return all_files

        except Exception as e:
            logger.error(f"Failed to get files: {str(e)}")
            raise RuntimeError(f"Failed to get files: {str(e)}") from e

    def _read_shard(self, path: str) -> list[dict[str, Any]]:
        """Read a single shard file."""
        with self.fs.open(path) as f:
            df = pd.read_parquet(io.BytesIO(f.read()))
        return cast(list[dict[str, Any]], df.to_dict(orient="records"))

    def _load_dataset_by_split(self, split) -> Union[Dataset, IterableDataset]:
        """Load dataset for a specific split."""
        if not self.source_config:
            raise ValueError("Source configuration is required to load dataset")
        sc = self.source_config
        ds = load_dataset(
            path=sc.repo_id,
            name=sc.config_name,
            split=split,
            streaming=sc.streaming,
            token=sc.token,
        )
        return cast(Union[Dataset, IterableDataset], ds)

    def _read_dataset(self) -> Union[list[dict[str, Any]], Iterator[dict[str, Any]]]:
        """Read complete dataset, handling multiple splits if specified."""
        try:
            if not self.source_config:
                raise ValueError("Source configuration is required to read dataset")
            sc = self.source_config
            if isinstance(sc.split, list):
                datasets_list = [self._load_dataset_by_split(split) for split in sc.splits]

                if len(datasets_list) == 1:
                    ds = datasets_list[0]
                else:
                    ds = concatenate_datasets(datasets_list)  # type: ignore[arg-type]
            else:
                ds = self._load_dataset_by_split(sc.split)

            if sc.streaming:
                return cast(Iterator[dict[str, Any]], ds)
            else:
                ds_concrete = cast(Dataset, ds)
                return cast(list[dict[str, Any]], ds_concrete.to_pandas().to_dict(orient="records"))

        except Exception as e:
            logger.error(f"Failed to read dataset: {str(e)}")
            raise RuntimeError(f"Failed to read dataset: {str(e)}") from e

    def _update_readme_config(self) -> None:
        """Update dataset card (README) with configuration details."""
        if not (
            self.output_config and self.output_config.repo_id and self.output_config.config_name
        ):
            return

        api = HfApi(token=hf_token)

        config_name = self.output_config.config_name
        split = self.output_config.split or "train"
        data_dir = config_name if config_name != "default" else "data"

        try:
            readme_path = api.hf_hub_download(
                repo_id=self.output_config.repo_id,
                filename=ds_config.REPOCARD_FILENAME,
                repo_type="dataset",
                token=self.output_config.token,
            )
            dataset_card = DatasetCard.load(readme_path)
            card_data = dataset_card.data
        except Exception:
            dataset_card = None
            card_data = DatasetCardData()

        metadata_configs = MetadataConfigs.from_dataset_card_data(card_data)

        if config_name not in metadata_configs:
            metadata_configs[config_name] = {}

        data_files_obj = metadata_configs[config_name].get("data_files", [])

        if isinstance(data_files_obj, str):
            data_files_obj = [{"split": split, "path": [data_files_obj]}]
        elif isinstance(data_files_obj, list):
            if all(isinstance(x, str) for x in data_files_obj):
                data_files_obj = [{"split": split, "path": data_files_obj}]
            elif all(isinstance(x, dict) for x in data_files_obj):
                pass
            else:
                new_lst = []
                for x in data_files_obj:
                    if isinstance(x, str):
                        new_lst.append({"split": split, "path": [x]})
                    elif isinstance(x, dict):
                        new_lst.append(x)
                data_files_obj = new_lst
        elif isinstance(data_files_obj, dict):
            tmp_list = []
            for s, paths in data_files_obj.items():
                if isinstance(paths, str):
                    paths = [paths]
                tmp_list.append({"split": s, "path": paths})
            data_files_obj = tmp_list
        else:
            data_files_obj = []

        new_pattern = f"{data_dir}/{split}-*"
        found_existing = False
        for entry in data_files_obj:
            if entry.get("split") == split:
                path_list = entry.setdefault("path", [])
                if new_pattern in path_list:
                    found_existing = True
                    break

        if found_existing:
            logger.info(
                f"README is already updated for config='{config_name}', "
                f"split='{split}', pattern='{new_pattern}'. Skipping commit."
            )
            return
        else:
            updated = False
            for entry in data_files_obj:
                if entry.get("split") == split:
                    path_list = entry.setdefault("path", [])
                    path_list.append(new_pattern)
                    updated = True
                    break
            if not updated:
                data_files_obj.append({"split": split, "path": [new_pattern]})

        metadata_configs[config_name]["data_files"] = data_files_obj
        metadata_configs.to_dataset_card_data(card_data)

        new_card_str = f"---\n{card_data}\n---\n"
        dataset_card = DatasetCard(new_card_str)

        operations = [
            CommitOperationAdd(
                path_in_repo=ds_config.REPOCARD_FILENAME,
                path_or_fileobj=new_card_str.encode("utf-8"),
            )
        ]
        commit_msg = (
            f"Add or update config='{config_name}', split='{split}' => pattern='{new_pattern}'"
        )
        api.create_commit(
            repo_id=self.output_config.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_msg,
            token=self.output_config.token,
        )
        logger.info(
            f"Updated README in {self.output_config.repo_id} for "
            f"config='{config_name}', split='{split}', pattern='{new_pattern}'."
        )
