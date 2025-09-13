import atexit
import hashlib
import json
import os
import signal
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Set

import datasets  # type: ignore[import-untyped]

from grasp.logger.logger_config import logger
from grasp.utils import constants, utils


class DatasetPositionTracker:
    """
    Abstract interface for tracking position in different types of datasets.
    """

    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type

    def advance_to_position(self, dataset, position: int):
        """Advance the dataset to a specific position."""
        raise NotImplementedError

    def get_current_position(self) -> int:
        """Get the current position in the dataset."""
        raise NotImplementedError

    def mark_position(self, position: int, record_id: Optional[str] = None):
        """Mark a position as processed."""
        raise NotImplementedError


class InMemoryPositionTracker(DatasetPositionTracker):
    """
    Tracker for in-memory datasets (lists, HF datasets).
    """

    def __init__(self):
        super().__init__("in_memory")
        self.current_position = 0
        self.highest_seen_position = 0  # Track highest position ever seen

    def advance_to_position(self, dataset, position: int):
        """For in-memory datasets, we can just slice the dataset."""
        if isinstance(dataset, list):
            return dataset[position:]
        elif isinstance(dataset, (datasets.Dataset, datasets.DatasetDict)):
            return dataset.skip(position)
        elif isinstance(dataset, datasets.IterableDataset):
            iterator = iter(dataset)
            iterator = iterator.skip(position)
            return iterator
        else:
            raise ValueError(f"Unsupported dataset type for in-memory tracking: {type(dataset)}")

    def get_current_position(self) -> int:
        return self.current_position

    def mark_position(self, position: int, record_id: Optional[str] = None):
        self.current_position = position
        if position > self.highest_seen_position:
            self.highest_seen_position = position


class StreamingPositionTracker(DatasetPositionTracker):
    """
    Tracker for streaming datasets that cannot be randomly accessed.
    Uses a window of recently seen records to detect duplicates.
    """

    def __init__(self, window_size: int = 1000):
        super().__init__("streaming")
        self.current_position = 0
        self.highest_seen_position = 0
        self.recent_ids: Deque[str] = deque(maxlen=window_size)
        self.position_to_record_id: Dict[int, str] = {}

    def advance_to_position(self, dataset, position: int):
        """
        For streaming datasets, we can't actually skip ahead.
        Instead, we'll need to process records and filter them out if already seen.
        """
        return dataset

    def get_current_position(self) -> int:
        return self.current_position

    def mark_position(self, position: int, record_id: Optional[str] = None):
        """
        Mark a position as processed, optionally with its record ID.
        This helps maintain the relationship between positions and records.
        """
        self.current_position = position
        if position > self.highest_seen_position:
            self.highest_seen_position = position
        if record_id:
            self.position_to_record_id[position] = record_id

    def record_seen(self, record_id: str) -> bool:
        """
        Check if a record has been seen recently and mark it as seen.
        Returns True if it was already seen.
        """
        if record_id in self.recent_ids:
            return True
        self.recent_ids.append(record_id)
        return False

    def get_record_id_for_position(self, position: int) -> Optional[str]:
        """Get the record ID associated with a specific position."""
        return self.position_to_record_id.get(position)


class ResumableExecutionManager:
    """
    Manages resumable execution with support for both in-memory and streaming datasets.
    """

    def __init__(self, task_name: str, output_file: str, window_size: int = 1000):
        """Initialize with task info and output file path."""
        self.task_name = task_name
        self.output_file = output_file
        self.output_dir = os.path.dirname(output_file)

        self.metadata_file = os.path.join(self.output_dir, "metadata.json")
        self.processed_records: Set[str] = set()
        self.in_process_records: Set[str] = set()
        self.last_save_time = time.time()
        self.save_interval = 10
        self.window_size = window_size
        self.metadata_initialized = False

        self.record_id_to_position: Dict[str, int] = {}

        self.total_records_processed = 0

        self.in_memory_tracker = InMemoryPositionTracker()
        self.streaming_tracker = StreamingPositionTracker(window_size)
        self.position_tracker: Optional[DatasetPositionTracker] = None

        self._setup_handlers()

    def _setup_handlers(self):
        """Set up signal and exit handlers to ensure metadata is saved."""
        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
            try:
                original_handler = signal.getsignal(sig)
                if hasattr(self, f"_original_handler_{sig}"):
                    continue

                setattr(self, f"_original_handler_{sig}", original_handler)

                signal.signal(sig, lambda s, f: self._signal_handler(s, f, original_handler))
            except (ValueError, OSError):
                pass

        atexit.register(self._exit_handler)

    def _signal_handler(self, signum, frame, original_handler=None):
        """Handle signals by saving state before exit."""
        logger.warning(f"Received signal {signum}, saving state before exit")
        self.force_save_state()

        if (
            original_handler
            and callable(original_handler)
            and original_handler not in (signal.SIG_DFL, signal.SIG_IGN)
        ):
            original_handler(signum, frame)
        else:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    def _exit_handler(self):
        """Handle normal program exit."""
        logger.info("Program exiting, saving final state")

        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r") as f:
                    metadata = json.load(f)
                    if metadata.get("completed", False):
                        logger.info("Execution already marked as complete, skipping final save")
                        return
            except Exception:
                pass

        self._do_save_state()

    @staticmethod
    def get_record_id(record: dict[str, Any]) -> str:
        """
        Compute a stable ID for a record.
        We check for explicit ID first, then compute a content hash if needed.
        """
        if "id" in record and record["id"]:
            return str(record["id"])

        record_str = json.dumps(record, sort_keys=True)
        return hashlib.sha256(record_str.encode()).hexdigest()

    def load_state(self, dataset_type: str = "auto") -> bool:
        """Load the execution state if it exists and is valid."""
        if not os.path.exists(self.metadata_file):
            logger.info(f"No metadata file found at {self.metadata_file}, starting fresh")
            return False

        try:
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Metadata file {self.metadata_file} is corrupted, starting fresh")
            return False

        # load the sampler pointers during resume of the process
        sampler_key_pointer = metadata.get(constants.META_SAMPLER_CACHE)
        # it can be None for old metadata
        if sampler_key_pointer:
            for k, v in sampler_key_pointer.items():
                logger.info(f"Loading sampler pointer for data source {k}: {v}")
                # set dataset object as None, this will be filled at utils.fetch_next_record()
                utils.sampler_cache[k] = (None, v)

        if metadata.get(constants.META_TASK_NAME) != self.task_name:
            logger.warning("Metadata file is for a different task, starting fresh")
            return False

        prev_output_file = metadata.get(constants.META_OUTPUT_FILE)

        if not prev_output_file or not os.path.exists(prev_output_file):
            base_path = os.path.splitext(prev_output_file)[0]
            alternatives = [
                f"{base_path}.json",
                f"{base_path}.jsonl",
            ]

            found_file = None
            for alt_file in alternatives:
                if os.path.exists(alt_file):
                    found_file = alt_file
                    break

            if not found_file:
                logger.warning(
                    f"Output file {prev_output_file} referenced in metadata doesn't exist and no alternatives found, starting fresh"
                )
                return False

            logger.info(f"Found alternative output file {found_file} instead of {prev_output_file}")
            prev_output_file = found_file

            metadata[constants.META_OUTPUT_FILE] = prev_output_file
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        if prev_output_file != self.output_file:
            logger.info(f"Output file has changed from {prev_output_file} to {self.output_file}")
            metadata[constants.META_OUTPUT_FILE] = self.output_file
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        self.processed_records = set(metadata.get(constants.META_PROCESSED_RECORDS, []))
        self.total_records_processed = len(self.processed_records)
        position = metadata.get(constants.META_LAST_POSITION, 0)

        if constants.META_RECORD_POSITIONS in metadata:
            self.record_id_to_position = metadata[constants.META_RECORD_POSITIONS]

        if dataset_type == "auto":
            dataset_type = metadata.get(constants.META_DATASET_TYPE, "in_memory")

        if dataset_type == "streaming":
            self.position_tracker = self.streaming_tracker

            if isinstance(self.position_tracker, StreamingPositionTracker):
                for record_id, pos in self.record_id_to_position.items():
                    self.position_tracker.position_to_record_id[int(pos)] = record_id
        else:
            self.position_tracker = self.in_memory_tracker

        self.position_tracker.mark_position(position)
        self.metadata_initialized = True

        logger.info(
            f"Loaded resume state with {len(self.processed_records)} processed records, "
            f"position {position}, using {dataset_type} mode"
        )
        return True

    def save_state(self) -> None:
        """Save the current execution state if enough time has passed."""
        current_time = time.time()
        if current_time - self.last_save_time < self.save_interval:
            return

        self._do_save_state()
        self.last_save_time = current_time

    def _mark_as_complete(self) -> None:
        """
        Mark the execution as complete. This updates the metadata file directly.
        """
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "r") as f:
                    metadata = json.load(f)
            else:
                metadata = self._build_current_metadata()

            metadata[constants.META_COMPLETED] = True
            metadata[constants.META_TIMESTAMP] = time.strftime(
                "%Y-%m-%d %H:%M:%S %z", time.localtime(time.time())
            )

            # Write the updated metadata to a temporary file
            temp_file = f"{self.metadata_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump(metadata, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Atomically rename to ensure consistency
            os.replace(temp_file, self.metadata_file)

            try:
                with open(self.metadata_file, "r") as f:
                    verified_metadata = json.load(f)
                    if not verified_metadata.get("completed", False):
                        logger.warning("Failed to verify completed status in metadata file")
                    else:
                        logger.info("Execution marked as complete and verified")
            except Exception as e:
                logger.warning(f"Could not verify metadata file after write: {e}")

        except Exception as e:
            logger.error(f"Error marking execution as complete: {e}")

    def _build_current_metadata(self) -> dict:
        """Build metadata from the current state."""
        if not self.position_tracker:
            position = 0
            highest_position = 0
        else:
            position = self.position_tracker.get_current_position()
            highest_position = getattr(self.position_tracker, "highest_seen_position", position)

        # Create record positions mapping
        record_positions = {}
        if hasattr(self, "record_id_to_position"):
            record_positions = self.record_id_to_position

        return {
            constants.META_TASK_NAME: self.task_name,
            constants.META_OUTPUT_FILE: self.output_file,
            constants.META_PROCESSED_RECORDS: list(self.processed_records),
            constants.META_LAST_POSITION: position,
            constants.META_HIGHEST_POSITION: highest_position,
            constants.META_DATASET_TYPE: (
                self.position_tracker.dataset_type if self.position_tracker else "in_memory"
            ),
            constants.META_TOTAL_PROCESSED: len(self.processed_records),
            constants.META_COMPLETED: False,  # Default to false, will be set to true in _mark_as_complete
            constants.META_RECORD_POSITIONS: record_positions,
            constants.META_TIMESTAMP: time.strftime(
                "%Y-%m-%d %H:%M:%S %z", time.localtime(time.time())
            ),
        }

    def force_save_state(self, is_final=False) -> None:
        """
        Force immediate save of the execution state.
        If is_final=True, marks as complete and ensures it's written to disk.

        Args:
            is_final: Whether this is the final save at the end of execution
        """
        if is_final:
            try:
                # initialize the sampler cache
                sampler_key_pointer = (
                    {k: v[1] for k, v in utils.sampler_cache.items()}
                    if len(utils.sampler_cache) > 0
                    else {}
                )

                if os.path.exists(self.metadata_file):
                    with open(self.metadata_file, "r") as f:
                        metadata = json.load(f)
                        # overwrite old pointer values
                        metadata[constants.META_SAMPLER_CACHE] = sampler_key_pointer
                else:
                    metadata = {
                        constants.META_TASK_NAME: self.task_name,
                        constants.META_OUTPUT_FILE: self.output_file,
                        constants.META_PROCESSED_RECORDS: list(self.processed_records),
                        constants.META_LAST_POSITION: (
                            self.position_tracker.get_current_position()
                            if self.position_tracker
                            else 0
                        ),
                        constants.META_HIGHEST_POSITION: (
                            getattr(self.position_tracker, "highest_seen_position", 0)
                            if self.position_tracker
                            else 0
                        ),
                        constants.META_DATASET_TYPE: (
                            self.position_tracker.dataset_type
                            if self.position_tracker
                            else "in_memory"
                        ),
                        constants.META_TOTAL_PROCESSED: len(self.processed_records),
                        constants.META_SAMPLER_CACHE: sampler_key_pointer,
                    }

                metadata[constants.META_COMPLETED] = True
                metadata[constants.META_TIMESTAMP] = time.strftime(
                    "%Y-%m-%d %H:%M:%S %z", time.localtime(time.time())
                )

                with open(self.metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                try:
                    with open(self.metadata_file, "r") as f:
                        check_metadata = json.load(f)
                        if not check_metadata.get(constants.META_COMPLETED, False):
                            logger.error("Failed to set completed flag in metadata file!")
                        else:
                            logger.info("Execution marked as complete and verified")
                except Exception as check_e:
                    logger.error(f"Error verifying completed flag: {check_e}")

            except Exception as e:
                logger.error(f"Error finalizing execution state: {e}")
        else:
            self._do_save_state()

    def _do_save_state(self) -> None:
        """Perform the actual state saving."""
        if not self.processed_records:
            return

        if not self.position_tracker:
            logger.warning("Cannot save state: no position tracker initialized")
            return

        position = self.position_tracker.get_current_position()
        highest_position = getattr(self.position_tracker, "highest_seen_position", position)

        record_positions = {}
        if isinstance(self.position_tracker, StreamingPositionTracker):
            for pos, record_id in self.position_tracker.position_to_record_id.items():
                if record_id in self.processed_records:
                    record_positions[record_id] = pos
        else:
            record_positions = self.record_id_to_position

        # initialize the sampler cache
        sampler_key_pointer = (
            {k: v[1] for k, v in utils.sampler_cache.items()}
            if len(utils.sampler_cache) > 0
            else {}
        )

        metadata = {
            constants.META_TASK_NAME: self.task_name,
            constants.META_OUTPUT_FILE: self.output_file,
            constants.META_PROCESSED_RECORDS: list(self.processed_records),
            constants.META_POSITION: position,
            constants.META_HIGHEST_POSITION: highest_position,
            constants.META_DATASET_TYPE: self.position_tracker.dataset_type,
            constants.META_TOTAL_PROCESSED: len(self.processed_records),
            constants.META_SAMPLER_CACHE: sampler_key_pointer,
            constants.META_COMPLETED: False,
            constants.META_RECORD_POSITIONS: record_positions,
            constants.META_TIMESTAMP: time.strftime(
                "%Y-%m-%d %H:%M:%S %z", time.localtime(self.last_save_time)
            ),
        }

        if not metadata[constants.META_PROCESSED_RECORDS]:
            logger.debug("No records processed yet, skipping metadata save")
            return

        if not os.path.exists(self.output_file):
            logger.debug(
                f"Output file {self.output_file} doesn't exist yet, skipping metadata save"
            )
            return

        try:
            temp_file = f"{self.metadata_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump(metadata, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_file, self.metadata_file)

            self.metadata_initialized = True
            logger.debug(f"Saved metadata to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def get_metadata(self) -> dict[str, Any]:
        """Get the current metadata state."""
        if not self.metadata_initialized:
            logger.warning("Metadata not initialized, returning empty state")
            return {}

        try:
            with open(self.metadata_file, "r") as f:
                metadata: dict[str, Any] = json.load(f)
                logger.info(f"Loaded metadata from {self.metadata_file}")
            return metadata

        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}

    def is_record_processed(self, record: dict[str, Any]) -> bool:
        """
        Check if a record has already been processed using its ID.
        For resumable execution, we rely on record ID to identify records.
        """
        record_id = self.get_record_id(record)

        if self.position_tracker and self.position_tracker.dataset_type == "streaming":
            # For streaming datasets, check both the processed records set and the recent window
            return record_id in self.processed_records or (
                hasattr(self.position_tracker, "record_seen")
                and self.position_tracker.record_seen(record_id)
            )

        return record_id in self.processed_records

    def mark_record_processing(self, record: dict[str, Any], position: int) -> None:
        """Mark a record as currently being processed (not yet complete)."""
        record_id = self.get_record_id(record)
        self.in_process_records.add(record_id)

        if self.position_tracker:
            # Pass the record_id to the tracker to maintain position-to-record mapping
            self.position_tracker.mark_position(position, record_id)

    def mark_record_processed(self, record: dict[str, Any], position: Optional[int] = None) -> None:
        """Mark a record as processed and optionally update position."""
        record_id = self.get_record_id(record)
        self.processed_records.add(record_id)
        self.in_process_records.discard(record_id)
        self.total_records_processed += 1

        if position is not None:
            self.record_id_to_position[record_id] = position

            if self.position_tracker:
                self.position_tracker.mark_position(position, record_id)

                if hasattr(self.position_tracker, "highest_seen_position"):
                    if position > self.position_tracker.highest_seen_position:
                        self.position_tracker.highest_seen_position = position

        if self.total_records_processed % 1 == 0:
            self.save_state()
