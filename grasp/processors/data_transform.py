"""Module for data transformation operations.

This module provides abstract and concrete classes for transforming data records.
Transformations can be applied to lists of dictionaries, allowing for data manipulation operations.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from grasp.logger.logger_config import logger
from grasp.utils.audio_utils import get_audio_fields, get_audio_url, load_audio
from grasp.utils.image_utils import (
    get_image_fields,
    get_image_url,
    is_data_url,
    load_image,
)


class DataTransform(ABC):
    """Abstract base class for data transformation operations.

    This class defines the interface for implementing data transformations
    that can be applied to lists of dictionary records.

    Each transformation must implement:
    1. name property: A unique identifier for the transform
    2. transform method: The actual transformation logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the transformation.

        Returns:
            str: Unique identifier for this transformation type.
        """
        pass

    @abstractmethod
    def transform(self, data: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        """Apply the transformation to a list of records.

        Args:
            data (list[dict[str, Any]]): List of dictionary records to transform.
            params (dict[str, Any]): Parameters controlling the transformation.

        Returns:
            list[dict[str, Any]]: Transformed list of dictionary records.
        """
        pass


class SkipRecords(DataTransform):
    """
    Skip records based on variables
    Example:
    - transform: grasp.processors.data_transform.SkipRecords
        params:
          skip_type: "range"/"count" (default:"range")
          range: "[:10],[-10:]"
          count:
              from_begin: 10
              from_end: 20
    """

    @property
    def name(self) -> str:
        """Get the name of the skip record transformation.

        Returns:
            str: The identifier 'skip_records'.
        """
        return "skip_records"

    def transform(self, data: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        """Skip records based on params.

        Args:
            data (list[dict[str, Any]]): List of dictionary records to transform.
            params (dict[str, Any]): Parameters containing:
                - skip_type: range or count (default:range)
                - range: range based python like syntax
                - count : Number of records to skip from begin and end using from_start/from_end key

        Returns:
            list[dict[str, Any]]: Transformed records.
        """
        # first read the dataset (needed for iterator/stream)
        dataset = [r for r in data]
        skip_type = params.get("skip_type", "range")
        ds_len = len(dataset)
        if skip_type == "count":
            skip_count = params.get("count", {})
            # skip number of records from beginning (useful during pdf ebooks)
            skip_begin = skip_count.get("from_start", 0)
            start = int(skip_begin)
            # skip number of records from end, also consider the combine count(as it cant happen in last element)
            skip_end = skip_count.get("from_end", 0)
            end = len(dataset) - int(skip_end)
            new_dataset = dataset[start:end]
        elif skip_type == "range":
            skip_range = params.get("range", "")
            ranges_str = skip_range.split(",")
            ranges = {}
            # build skip ranges from string parsing
            for r in ranges_str:
                res = re.findall(r"\[(.*)]", r)
                if len(res) == 0:
                    continue
                rng = res[0].split(":")
                start = 0 if len(rng[0]) == 0 else int(rng[0])
                end = ds_len if len(rng[1]) == 0 else int(rng[1])
                if start < 0:
                    start = ds_len + start
                if end < 0:
                    end = ds_len + end
                ranges[start] = end

            # build a set of skip indices
            skip_indices: set[int] = set()
            for s, e in ranges.items():
                skip_indices.update(range(s, e))
            # build new dataset by skipping the data
            new_dataset = [record for i, record in enumerate(dataset) if i not in skip_indices]
        else:
            raise Exception(f"Unknown skip type '{skip_type}'")

        # return the newly created dataset
        return new_dataset


class CombineRecords(DataTransform):
    """
    Combine records based on variables
    Example:
    - transform: grasp.processors.data_transform.CombineRecords
        params:
          skip:                        (Skip 10 records from front and end of dataset)
            from_beginning: 10
            from_end: 10
          combine: 2                   (Combine 2 records, current with next)
          shift: 1                     (iterate to next record after combining)
          join_column:                 (column join rules)
            page: "$1-$2"              (combine page column from record 1 and 2 with a dash)
            pdf_reader: "$1\n\n$2"     (combine pdf_reader column from record 1 and 2 with 2 new lines)
            llm_extract: "$1\n\n$2"    (combine llm_Extract column from record 1 and 2 with 2 new lines)
            type: "$1"                 (for type column, do not combine, just pick from record 1)
            model: "$1"                (for model column, do not combine, just pick from record 1)
            metadata: "$1"             (for metadata column, do not combine, just pick from record 1)
    """

    @property
    def name(self) -> str:
        """Get the name of the combine record transformation.

        Returns:
            str: The identifier 'combine_records'.
        """
        return "combine_records"

    def _replace_build_data(self, replace_string: str, records: list, col: str):
        """
        Fetch column data from records and join using the replace_string
        It will have variables like $1, $2, $3, $4...
        Replace variables with actual values of the column(col) from each record
        Example: "$1---$2" => "records[0][col]----records[1][col]"
        """
        var_count = len(records)
        # if len(re.findall("\$\d+", replace_string))  !=  var_count:
        #    raise Exception(f"Invalid replace string: {replace_string}")
        for i in range(var_count):
            # note: replace variable index starts from 1
            record_index = i + 1
            replace_string = replace_string.replace(f"${record_index}", str(records[i][col]))
        return replace_string

    def transform(self, data: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        """Combine records based on params.

        Args:
            data (list[dict[str, Any]]): List of dictionary records to transform.
            params (dict[str, Any]): Parameters containing:
                - skip.from_beginning: skip number of records from beginning, needed specially to skip introduction pages in an ebook
                - skip.from_end: skip number of records from end, needed to skip index pages from an ebook
                - combine : Number of records to combine, 1 means no records to combine
                - shift : After combining, shift by how many records. Similar to stride in CNN convolution
                - join_column: a dictionary of column name and replace string($1 represents data from 1st record and so on)
                      id : "$1-$2"  it means join 2 ids with a dash

        Returns:
            list[dict[str, Any]]: Transformed records.
        """
        # first read the dataset (needed for iterator/stream)
        dataset = [r for r in data]
        # how many records to combine, 2 means current plus next, 0 or 1 means nothing to combine(return as it is)
        combine = int(params.get("combine", 1))
        if combine < 2:
            return dataset
        # if shift is 2, after combining (1,2,3), it will do (3,4,5), next (5,6,7)
        shift = int(params.get("shift", 1))
        # skip number of records from beginning (useful during pdf ebooks)
        skip_begin = params.get("skip", {}).get("from_beginning", 0)
        start = int(skip_begin)
        # skip number of records from end, also consider the combine count(as it cant happen in last element)
        skip_end = params.get("skip", {}).get("from_end", 0)
        end = len(dataset) - int(skip_end) - combine
        # columns to join and corresponding regex string
        join_column = params.get("join_column", {})
        # storage for new dataset
        new_dataset = []
        # iterate the original dataset starting from 'start' till 'end', by moving 'shift'
        for i in range(start, end, shift):
            # get records to join
            records = dataset[i : i + combine]
            # storage for the new record
            new_record = {}
            # fetch each column rule, execute and save
            for col, rule in join_column.items():
                new_record[col] = self._replace_build_data(rule, records, col)
            # finally store the new record
            new_dataset.append(new_record)
        # return the newly created dataset
        return new_dataset


class RenameFieldsTransform(DataTransform):
    """Transformer for renaming fields in dictionary records.

    This transformer allows renaming of dictionary keys based on a provided mapping.
    It can handle both simple renames and conditional renames based on overwrite settings.
    """

    @property
    def name(self) -> str:
        """Get the name of the rename fields transformation.

        Returns:
            str: The identifier 'rename_fields'.
        """
        return "rename_fields"

    def transform(self, data: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        """Rename fields in each record according to the provided mapping.

        Args:
            data (list[dict[str, Any]]): List of dictionary records to transform.
            params (dict[str, Any]): Parameters containing:
                - mapping (dict[str, str]): Old field name to new field name mapping
                - overwrite (bool): Whether to overwrite existing fields

        Returns:
            list[dict[str, Any]]: Transformed records with renamed fields.
        """
        return [self._rename_fields(record, params) for record in data]

    @staticmethod
    def _rename_fields(record: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Rename fields in a single record.

        Args:
            record (dict[str, Any]): Dictionary record to transform.
            params (dict[str, Any]): Transformation parameters.

        Returns:
            dict[str, Any]: Transformed record with renamed fields.
        """
        new_record = record.copy()
        mapping = params.get("mapping", {})
        overwrite = params.get("overwrite", False)

        for old_key, new_key in mapping.items():
            if old_key in new_record:
                if not overwrite and new_key in new_record:
                    continue
                new_record[new_key] = new_record.pop(old_key)
        return new_record


class AddNewFieldTransform(DataTransform):
    """Transformer for adding new fields to dictionary records.

    This transformer allows adding new fields to each record based on a provided mapping.
    It can handle both static values and dynamic values based on existing fields.
    """

    @property
    def name(self) -> str:
        """Get the name of the add new field transformation.

        Returns:
            str: The identifier 'add_new_field'.
        """
        return "add_new_field"

    def transform(self, data: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        """Add new fields to each record according to the provided mapping.

        Args:
            data (list[dict[str, Any]]): List of dictionary records to transform.
            params (dict[str, Any]): Parameters containing:
                - mapping (dict[str, str]): New field name to value mapping

        Returns:
            list[dict[str, Any]]: Transformed records with added fields.
        """
        return [self._add_new_fields(record, params) for record in data]

    def _add_new_fields(self, record: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Add new fields to a single record.

        Args:
            record (dict[str, Any]): Dictionary record to transform.
            params (dict[str, Any]): Transformation parameters.

        Returns:
            dict[str, Any]: Transformed record with added fields.
        """
        new_record = record.copy()
        mapping = params.get("mapping", {})

        for field_name, value in mapping.items():
            new_record[field_name] = value

        return new_record


class CreateImageUrlTransform(DataTransform):
    @property
    def name(self) -> str:
        """Get the name of the create image URL transformation.
        Returns:
            str: The identifier 'create_image_url'.
        """
        return "create_image_url"

    def transform(self, data: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        """Transform image fields in each record to base64-encoded data URLs.
        Args:
            data (list[dict[str, Any]]): List of dictionary records to transform.
            params (dict[str, Any]): Parameters controlling the transformation.
        Returns:
            list[dict[str, Any]]: Transformed records with image URLs.
        """
        if not data:
            logger.warning("No data provided for image URL transformation.")
            return data

        image_fields = get_image_fields(data[0])

        for record in data:
            for field in image_fields:
                image_data = record.get(field)
                if image_data is None:
                    continue
                record[field] = self.process_image_data(image_data)

        return data

    def process_image_data(self, image_data: Any) -> Any:
        """Process and convert image(s) to base64-encoded data URLs."""

        if isinstance(image_data, list):
            result = []
            for item in image_data:
                if is_data_url(item):
                    result.append(item)
                else:
                    try:
                        img = load_image(item)
                        result.append(get_image_url(img) if img else None)
                    except Exception as e:
                        logger.warning(f"Failed to process image in list: {e}")
                        result.append(item)
            return result

        if is_data_url(image_data):
            return image_data

        try:
            img = load_image(image_data)
            return get_image_url(img) if img else None
        except Exception as e:
            logger.warning(f"Failed to process image: {e}")
            return None


class CreateAudioUrlTransform(DataTransform):
    """DataTransform that replaces audio fields with base64-data URLs."""

    @property
    def name(self) -> str:
        return "create_audio_url"

    def transform(
        self,
        data: list[dict[str, Any]],
        params: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        if not data:
            logger.warning("No data provided to CreateAudioUrlTransform")
            return data

        params = params or {}
        output_field_map = params.get("output_fields", {})  # e.g., { "audio": "audio_base64" }
        audio_fields = get_audio_fields(data[0])

        for record in data:
            record_id = record.get("id", "<no-id>")
            for field in audio_fields:
                raw = record.get(field)
                if raw is None:
                    logger.debug(f"Record {record_id}: No data in field '{field}'")
                    continue

                processed = self._process_field(raw, field)
                if processed is not None:
                    out_field = output_field_map.get(field, field)
                    record[out_field] = processed
                else:
                    logger.warning(f"Record {record_id}: Failed to process audio field '{field}'")

        return data

    def _process_field(self, value: Any, field: str) -> Any:
        """Handle list or single-item audio values."""
        if isinstance(value, list):
            return [self._process_single(item, field) for item in value if item is not None]
        return self._process_single(value, field)

    def _process_single(self, item: Any, field: str) -> Union[str, None]:
        """Convert one audio-like item to a base64 data URL."""
        if not item:
            return None

        # Already a data URL?
        if isinstance(item, str) and is_data_url(item):
            return item

        try:
            audio_bytes = load_audio(item)
        except Exception as e:
            logger.exception(f"Exception while loading audio in field '{field}': {e}")
            return None

        if not audio_bytes:
            logger.warning(f"Could not load audio for item: {item!r}")
            return None

        mime = self._guess_mime(item)
        url = get_audio_url(audio_bytes, mime=mime)
        return url

    def _guess_mime(self, item: Any) -> str:
        """Guess audio MIME type based on known file extensions only."""
        if isinstance(item, str):
            ext = os.path.splitext(item)[1].lower()
            if ext == ".mp3":
                return "audio/mpeg"
            elif ext == ".ogg":
                return "audio/ogg"
            elif ext == ".flac":
                return "audio/flac"
            elif ext == ".aac":
                return "audio/aac"
            elif ext == ".m4a":
                return "audio/mp4"
            elif ext == ".aiff":
                return "audio/aiff"
            elif ext == ".wav":
                return "audio/wav"

        return "audio/wav"  # Default fallback
