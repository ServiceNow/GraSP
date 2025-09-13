from typing import Any

from pydantic import ValidationError

from grasp.data_mapper.helper import PipelineConfig, transform_registry
from grasp.data_mapper.pipelines import PipelineFactory
from grasp.logger.logger_config import logger
from grasp.validators.custom_schemas import CoreLLMDataFabricFormat, MetaInfo


class DataMapper:
    """Maps data items through a transformation pipeline based on the graph configuration."""

    def __init__(self, config: dict[str, Any]):
        """
        Initializes DataMapper with graph configuration and sets up the transformation pipeline.

        Args:
            graph_config (GraphConfig): The configuration for the graph transformation.
        """
        self.config = config
        self.transform_registry = transform_registry  # Registry for managing transformations
        # Normalize and validate transformation type (expects 'sft' or 'dpo')
        self.transform_type: str = str(config.get("type", ""))
        if not self.transform_type:
            raise ValueError(
                "Transform type must be provided in config['type'] (e.g., 'sft' or 'dpo')."
            )

        # Create the pipeline using PipelineFactory based on the transform type
        pipeline_factory = PipelineFactory(self.transform_type)
        pipeline = pipeline_factory.get_pipeline()

        # Create the PipelineConfig object
        self.pipeline_config = PipelineConfig(pipeline=pipeline)

        self.pipeline = self.pipeline_config.pipeline

    def map_single_item(self, old_item: dict[str, Any]) -> list[dict[str, Any]]:
        """Map a single data item through the transform pipeline."""
        try:
            # Setting up the context for transformation
            context: dict[str, Any] = {}
            context["__old_item__"] = old_item
            # logger.info(f"Initial context setup for item {old_item.get('id')}: {context}")

            # Update the context with additional fields like source_id, tags, etc.
            context.update(
                {
                    "source_id": old_item.get("id"),
                    "annotation_type": old_item.get("annotation_type", []),
                    "tags": old_item.get("tags", []),
                }
            )
            # logger.info(f"Updated context with source data: {context}")

            # Order the pipeline transforms based on their dependencies
            ordered_transforms = self.order_pipeline()
            logger.info(f"Ordered transforms: {ordered_transforms}")

            # Apply each transform in the ordered pipeline
            for transform in ordered_transforms:
                try:
                    # Get the current pipeline step
                    step = next(s for s in self.pipeline if s.name == transform.meta.name)
                    logger.info(f"Processing step: {step}")

                    # Validate that all required context fields are available for the current transform
                    self.transform_registry.validate_requirements(transform, context)
                    logger.info(f"Validated requirements for transform '{transform.meta.name}'")

                    # Set the current step in context for logging and tracking
                    context["__current_step__"] = {
                        "step": step,
                        "old_key": step.old_key,
                        "new_key": step.new_key,
                    }

                    # Extract the value from the old item for transformation based on the step's old_key
                    value = old_item.get(step.old_key) if step.old_key else None
                    logger.debug(f"Extracted value for {step.old_key}: {value}")

                    # Perform the transformation for the current step
                    try:
                        transform.transform(value, context)
                        logger.debug(
                            f"Successfully applied transform '{transform.meta.name}' to value: {value}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error in transform '{transform.meta.name}' with old_key='{step.old_key}': {str(e)}. Please ensure input data is in conversation format. If not, disable OASST mapping accordingly."
                        )
                        continue  # Skip this step and proceed to the next one
                    finally:
                        # Clean up the current step from context after the transformation
                        context.pop("__current_step__", None)

                except Exception as e:
                    # Handle unexpected errors during the transform step processing
                    logger.error(
                        f"Unexpected error during processing of step '{transform.meta.name}': {str(e)}"
                    )
                    continue  # Skip to the next transform if something unexpected happens

            # After all transforms are applied, build the rows from the final context
            logger.info("Finished all transforms, moving to building rows using context")
            # Build rows at the end of the process
            result = self.build_rows_and_validate(context)
            logger.debug(f"Finished building rows : {result}")
            logger.info("Rows built and validated against CoreLLMDataFabricFormat successfully")
            return result

        except Exception as e:
            # Handle any error during the entire process of mapping a single item
            logger.error(f"Error in mapping single item {old_item.get('id')}: {str(e)}")
            return []  # Return an empty list if an error occurs while processing the item

    def map_all_items(self, old_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Map all items through the transform pipeline"""

        logger.info(f"Starting {self.transform_type} transformation")

        # Ensure the data is in list format
        if not isinstance(old_data, list):
            old_data = list(old_data)
        all_rows = []
        for item in old_data:
            try:
                # Map each individual item and collect the results
                rows = self.map_single_item(item)
                all_rows.extend(rows)
            except Exception as e:
                # Log errors for each item processed
                logger.error(f"Error processing item: {item}: {str(e)}")
                continue
        return all_rows

    def order_pipeline(self, active=True) -> list:
        """Automatically order transforms based on their dependencies."""
        if not active:
            return self.pipeline

        # Fetch the transform objects from the registry
        transforms = [self.transform_registry.get(step.name) for step in self.pipeline]
        ordered = []
        provides_map = {}

        # Build a map of fields that each transform provides
        for transform in transforms:
            for field in transform.meta.provides:
                provides_map[field] = transform

        # Order the transforms by their requirements and dependencies
        for transform in transforms:
            for req in transform.meta.requires:
                provider = provides_map.get(req)
                if provider and provider not in ordered:
                    ordered.append(provider)
            if transform not in ordered:  # Avoid duplicates in the ordered list
                ordered.append(transform)

        return ordered

    @staticmethod
    def build_rows_and_validate(context: dict[str, Any]) -> list[dict[str, Any]]:
        """Build final rows using accumulated context and validate them."""

        # Extract necessary fields from the context
        conversation_id = context["conversation_id"]
        root_message_id = context["root_message_id"]
        messages = context["messages"]

        # Optional fields from the context
        categories = context.get("categories", [])
        subcategories = context.get("subcategories", [])
        languages = context.get("language", ["en"])
        source_id = context.get("source_id")
        annotation_type = (
            context.get("annotation_type", []) if context.get("annotation_type") else []
        )
        generated_by = context.get("generated_by", "grasp")
        tags = context.get("tags", [])

        rows = []
        for msg in messages:
            # Create the MetaInfo and CoreLLMDataFabricFormat for each message
            metainfo = MetaInfo(
                source_id=str(source_id) if source_id else None,
                source_metadata={"annotation_type": annotation_type},
            )

            row_model = CoreLLMDataFabricFormat(
                conversation_id=conversation_id,
                message_id=msg["message_id"],
                parent_id=msg["parent_id"],
                root_message_id=root_message_id,
                message_level=msg["level"],
                role=msg["role"],
                content=msg["content"],
                languages=languages,
                categories=categories,
                subcategories=subcategories,
                instruction_tags=msg.get("instruction_tags", []),
                generated_by=generated_by,
                metainfo=metainfo,
                quality=msg.get("quality", {}),
                tags=tags,
                length=msg.get("length", {}),
                data_characteristics=msg.get("data_characteristics", {}),
            )

            # Validate the row and handle any validation errors
            try:
                CoreLLMDataFabricFormat(**row_model.model_dump())
                rows.append(row_model.model_dump())
            except ValidationError as e:
                logger.warning(f"Validation error: {e}. Skipping row.")
        return rows
