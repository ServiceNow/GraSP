import asyncio
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, Union, cast

import datasets  # type: ignore[import-untyped]
import tqdm  # type: ignore[import-untyped]
from langgraph.graph.state import CompiledStateGraph

from sygra.core.graph.graph_config import GraphConfig
from sygra.core.resumable_execution import ResumableExecutionManager
from sygra.data_mapper.mapper import DataMapper
from sygra.logger.logger_config import logger
from sygra.utils import constants, graph_utils, multimodal_processor, utils
from sygra.validators.schema_validator_base import SchemaValidator


class DatasetProcessor:
    def __init__(
        self,
        input_dataset: Union[list[dict], datasets.Dataset, datasets.IterableDataset],
        graph: CompiledStateGraph,
        graph_config: GraphConfig,
        output_file: str,
        num_records_total: int,  # can't use len(input_dataset) as it is not supported for IterableDataset
        start_index: int = 0,  # used to initialize the dataset_indx
        batch_size: int = 50,
        checkpoint_interval: int = 100,
        debug: bool = False,
        input_record_generator: Optional[Callable] = None,
        output_record_generator: Optional[Callable] = None,
        resumable: bool = False,
        task_name: Optional[str] = None,
    ):
        assert (
            checkpoint_interval % batch_size == 0
        ), "Checkpoint Interval should be a multiple of Batch Size"

        # capture the input parameters
        self.original_dataset = input_dataset
        self.graph = graph
        self.graph_config = graph_config
        self.output_file = output_file
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.debug = debug
        self.input_record_generator = input_record_generator
        self.output_record_generator = output_record_generator
        self.is_valid_schema = True

        # initialize the state variables
        self.dataset_indx = start_index
        self.start_time = time.time()
        self.batch_start = time.time()
        self.num_records_processed = 0
        self.failed_records = 0

        # Resumable execution settings
        self.resumable = resumable
        self.resume_manager: Optional[ResumableExecutionManager] = None
        self.task_name = task_name or (
            graph_config.config.get("task_name", self._extract_task_name())
            if hasattr(graph_config, "config")
            else self._extract_task_name()
        )

        # Determine actual number of records to process (after accounting for already processed records)
        self.num_records_total = num_records_total

        # Is this a streaming dataset?
        self.is_streaming = not isinstance(input_dataset, list)

        # Setup resumable execution if enabled - MUST be called before initializing progress bar
        if self.resumable:
            self._setup_resumable_execution()

        # Initialize the tqdm progress bar with the correct number of records to process
        self.pbar = tqdm.tqdm(total=self.num_records_total)
        self.graph_results: list[dict[str, Any]] = []

        # Initialize input dataset iterator
        self.input_dataset = iter(input_dataset)

    def _extract_task_name(self) -> str:
        """
        Extract task name from output file path as a fallback when not provided.
        Ensures we have a valid task identifier for resumable execution.
        """
        try:
            path_parts = self.output_file.split("/")
            for i, part in enumerate(path_parts):
                if part == "tasks" and i + 1 < len(path_parts):
                    return path_parts[i + 1]

            return f"task_{hash(self.output_file) % 10000:04d}"
        except Exception:
            return f"task_{uuid.uuid4().hex[:8]}"

    def _setup_resumable_execution(self):
        """Initialize the resumable execution manager and load previous state if available."""
        self.resume_manager = ResumableExecutionManager(
            task_name=self.task_name,
            output_file=self.output_file,
        )

        dataset_type = self._determine_dataset_type(self.original_dataset)

        if dataset_type == "streaming":
            self.resume_manager.position_tracker = self.resume_manager.streaming_tracker
        else:
            self.resume_manager.position_tracker = self.resume_manager.in_memory_tracker

        if self.resume_manager.load_state(dataset_type):
            processed_count = len(self.resume_manager.processed_records)
            self.num_records_processed = processed_count
            logger.info(f"Resuming execution: {processed_count} records already processed")
        else:
            logger.info("No previous state found or resumable execution disabled. Starting fresh.")
            self.resume_manager.position_tracker.mark_position(self.dataset_indx)

    @staticmethod
    def _determine_dataset_type(dataset) -> str:
        """Determine if the dataset is in-memory or streaming."""
        if isinstance(dataset, list):
            return "in_memory"
        elif hasattr(dataset, "is_streaming") and dataset.is_streaming:
            return "streaming"
        elif isinstance(dataset, datasets.IterableDataset):
            return "streaming"
        # Default to in-memory for safety
        return "in_memory"

    def _get_record(self) -> Optional[dict[str, Any]]:
        """
        Get the next record from the dataset, handling resumable execution by skipping processed records.

        Returns:
            A dictionary containing the next unprocessed record

        Raises:
            StopIteration: When no more unprocessed records are available
        """
        while True:  # Keep trying until we get an unprocessed record or exhaust the dataset
            # Get next record from dataset
            record = next(self.input_dataset)

            # Ensure record has an ID
            if "id" not in record or not record["id"]:
                record["id"] = str(uuid.uuid4())

            # For resumable execution, skip already processed records
            if self.resumable and self.resume_manager:
                record_id = self.resume_manager.get_record_id(record)

                if self.resume_manager.is_record_processed(record):
                    # Skip this record and get another
                    logger.debug(f"Skipping already processed record: {record_id}")
                    self.dataset_indx += 1
                    continue

                # Mark the record as being processed (not yet completed)
                self.resume_manager.mark_record_processing(record, self.dataset_indx)

            # This is a record we need to process - increment index and return
            self.dataset_indx += 1
            return record

    def process_and_store_results(self):
        """Main entry point to process the dataset and store results."""
        asyncio.run(self._process_and_store_results())

    @staticmethod
    def is_error_code_in_output(output_record: dict[str, Any]) -> bool:
        """
        Check if the output record contains an error code.

        Args:
            output_record: The output record to check

        Returns:
            True if an error code is found, False otherwise
        """
        for key, value in output_record.items():
            if isinstance(value, str) and value.startswith(constants.ERROR_PREFIX):
                return True
        return False

    async def _add_graph_result(self, output: dict[str, Any], record: dict[str, Any]) -> None:
        """
        Add a result from graph execution to the results list and handle resumable state.

        Args:
            output: The output from graph execution
            record: The input record that was processed
        """
        # Check if execution had an error - don't mark as processed if it did
        if self.is_error_code_in_output(output):
            self.failed_records += 1
            # For resumable execution, remove from in-process but don't mark as processed
            if self.resumable and self.resume_manager:
                logger.warning(
                    f"Execution error detected for record {record.get('id')}. Will retry on next run."
                )
                record_id = self.resume_manager.get_record_id(record)
                self.resume_manager.in_process_records.discard(record_id)
            else:
                logger.error(
                    f"Execution error detected for record {record.get('id')}. Skipping record."
                )
            return

        # Schema validation for custom output schema runs when oasst_mapper is set to false
        is_oasst_mapper_required = (
            True
            if (
                isinstance(self.graph_config.oasst_mapper, dict)
                and self.graph_config.oasst_mapper.get("required") == "yes"
            )
            else False
        )

        # Validate schema if needed
        if not is_oasst_mapper_required:
            logger.info("Starting custom schema validation")
            schema_validator = SchemaValidator(self.graph_config)
            self.is_valid_schema = schema_validator.validate(output)
            if not self.is_valid_schema:
                logger.error("Output data validation failed, skipping record")
                if self.resumable and self.resume_manager:
                    record_id = self.resume_manager.get_record_id(record)
                    self.resume_manager.in_process_records.discard(record_id)
                return

        # Add result to batch
        self.graph_results.append(output)
        self.num_records_processed += 1

        # all the code below should refer total_records_with_error(not self.num_records_processed)
        total_records_with_error = self.num_records_processed + self.failed_records

        if total_records_with_error >= self.num_records_total:
            logger.info(f"Reached target of {self.num_records_total} records. Stopping.")

        # For resumable execution, mark record as fully processed
        if self.resumable and self.resume_manager:
            # Set the current dataset position when marking the record as processed
            self.resume_manager.mark_record_processed(record, self.dataset_indx - 1)

            # Force an immediate state save every few records
            if total_records_with_error % self.batch_size == 0:
                self.resume_manager.force_save_state()

        # Log progress periodically
        if total_records_with_error % self.batch_size == 0:
            logger.info(
                f"Processed {total_records_with_error} out of {self.num_records_total} records "
                f"in {(time.time() - self.start_time):0.2f} secs; "
                f"Avg time per record: {((time.time() - self.start_time) / total_records_with_error):0.2f} secs; "
                f"Avg time per record for last batch: {((time.time() - self.batch_start) / len(self.graph_results)):0.2f} secs"
            )
            self.batch_start = time.time()

        # Write checkpoint at regular intervals
        if (
            total_records_with_error % self.checkpoint_interval == 0
            or total_records_with_error == self.num_records_total
        ):
            logger.info(
                f"Writing checkpoint with {self.num_records_processed} successful records and skipping {self.failed_records} failed records."
            )
            await self._write_checkpoint(is_oasst_mapper_required)

    async def _write_checkpoint(self, is_oasst_mapper_required: bool) -> None:
        """
        Write the current results to the output file and save resumable state.

        Args:
            is_oasst_mapper_required: Whether OASST mapping is required
        """
        file_write_start = time.time()

        # Convert graph outputs to records
        output_records = graph_utils.convert_graph_output_to_records(
            self.graph_results, self.output_record_generator
        )

        # Process multimodal data: save base64 data URLs to files and replace with file paths
        try:
            multimodal_output_dir = ".".join(self.output_file.split(".")[:-1])
            output_records = multimodal_processor.process_batch_multimodal_data(
                output_records, Path(multimodal_output_dir)
            )
        except Exception as e:
            logger.warning(f"Failed to process multimodal data: {e}. Continuing with original records.")

        # Handle intermediate writing if needed
        if (
            is_oasst_mapper_required
            and isinstance(self.graph_config.oasst_mapper, dict)
            and self.graph_config.oasst_mapper.get("intermediate_writing") == "yes"
        ):
            intermediate_write_path = (
                ".".join(self.output_file.split(".")[:-1])
                + constants.INTERMEDIATE
                + self.output_file.split(".")[-1]
            )
            logger.info(f"Writing intermediate file: {intermediate_write_path}")
            if ".jsonl" in self.output_file:
                utils.append_to_jsonl_file(intermediate_write_path, output_records)
            else:
                utils.append_to_json_file(intermediate_write_path, output_records)

        # Apply OASST mapping if required
        if is_oasst_mapper_required:
            try:
                mapper_cfg = cast(dict[str, Any], self.graph_config.oasst_mapper)
                mapper = DataMapper(config=mapper_cfg)
                oasst_mapped_output = mapper.map_all_items(output_records)
            except Exception as e:
                logger.error(f"Failed to apply oasst_mapper with error: {e}")
                oasst_mapped_output = output_records
        else:
            logger.info("OASST Mapping has been disabled. Skipping OASST mapping.")
            oasst_mapped_output = output_records

        # Write to output file
        if ".jsonl" in self.output_file:
            utils.append_to_jsonl_file(self.output_file, oasst_mapped_output)
        else:
            utils.append_to_json_file(self.output_file, oasst_mapped_output)

        logger.info(
            f"Updated {self.output_file} with the latest {len(self.graph_results)} records "
            f"in {(time.time() - file_write_start):0.2f} secs"
        )

        # Clear the processed results after writing to the file
        self.graph_results = []

        # Force save the resume state if enabled
        if self.resumable and self.resume_manager:
            self.resume_manager.force_save_state()

    def _handle_signal(self, signum, frame):
        """
        Handle signals by ensuring state is saved before shutdown.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.warning(f"Received signal {signum}, initiating graceful shutdown")

        if self.resumable and self.resume_manager:
            logger.info("Saving resume state before shutdown")
            self.resume_manager.force_save_state()

        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    async def _process_and_store_results(self):
        """Main processing loop for dataset records."""
        # Register signal handlers for graceful shutdown if resumable
        if self.resumable and self.resume_manager:
            for sig in [signal.SIGTERM, signal.SIGINT]:
                signal.signal(sig, lambda s, f: self._handle_signal(s, f))

        # Use asyncio tasks to process records in parallel up to batch_size
        pending_tasks: set[asyncio.Task] = set()
        # Track how many records we've started processing to limit total for streaming datasets
        records_started = 0

        try:
            while True:
                # Fill the pending tasks pool up to batch_size, but only if we haven't
                # already started processing our target number of records
                while (
                    len(pending_tasks) < self.batch_size
                    and records_started < self.num_records_total
                ):
                    try:
                        record = self._get_record()
                        task = asyncio.create_task(self._process_record(record))
                        pending_tasks.add(task)
                        task.add_done_callback(
                            lambda t: pending_tasks.discard(t) if t in pending_tasks else None
                        )
                        # Increment count of records we've started processing
                        records_started += 1
                    except StopIteration:
                        # No more records to process
                        break

                # Exit loop if no pending tasks
                if not pending_tasks:
                    break

                # Wait for at least one task to complete
                done, pending_tasks = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                # Update progress bar for completed tasks
                for task in done:
                    try:
                        task.result()  # This will raise any exceptions from the task
                        self.pbar.update(1)
                    except Exception as e:
                        logger.error(f"Task failed with error: {e}")

        except Exception as e:
            logger.error(f"Error in main processing loop: {e}")

        finally:
            # Close the progress bar
            self.pbar.close()

            # Save final state for resumable execution
            if self.resumable and self.resume_manager:
                logger.info("Saving final execution state")
                self.resume_manager.force_save_state()

                # Write any remaining results
                if self.graph_results:
                    logger.info(f"Writing {len(self.graph_results)} remaining results")
                    is_oasst_mapper_required = (
                        self.graph_config.oasst_mapper is not None
                        and self.graph_config.oasst_mapper.get("required") == "yes"
                    )
                    await self._write_checkpoint(is_oasst_mapper_required)

    async def _process_record(self, record: Optional[dict[str, Any]]) -> None:
        """
        Process a single record through the graph.

        Args:
            record: Input record to process
        """
        try:
            if record is None:
                logger.warning("Received None record, skipping")
                return
            if self.num_records_processed >= self.num_records_total:
                logger.debug(
                    f"Already processed target number of records. Skipping record {record.get('id')}"
                )
                return

            graph_result = await graph_utils.execute_graph(
                record,
                self.graph,
                debug=self.debug,
                input_record_generator=self.input_record_generator,
            )

            if self.num_records_processed < self.num_records_total:
                await self._add_graph_result(graph_result, record)

        except Exception as e:
            rec_id_str = "unknown"
            if isinstance(record, dict):
                rec_id_str = str(record.get("id", "unknown"))
            logger.error(f"Error processing record {rec_id_str}: {e}")

            if self.resumable and self.resume_manager:
                safe_record: dict[str, Any] = record if isinstance(record, dict) else {}
                record_id = self.resume_manager.get_record_id(safe_record)
                self.resume_manager.in_process_records.discard(record_id)
                if self.resume_manager.position_tracker is not None:
                    self.resume_manager.position_tracker.mark_position(self.dataset_indx)
                self.resume_manager.save_state()
