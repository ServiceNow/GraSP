from grasp.validators.custom_schemas import PipelineStep


class SFTPipelineBuilder:
    """Builds the pipeline steps for 'sft' transformation."""

    def __init__(self):
        self.pipeline_steps = []

    def build_pipeline(self) -> list[PipelineStep]:
        """Build and return the pipeline for 'sft' transformation."""
        pipeline_steps = [
            PipelineStep(name="copy", old_key="category", new_key="categories"),
            PipelineStep(name="copy", old_key="subcategory", new_key="subcategories"),
            PipelineStep(name="taxonomy", old_key="taxonomy", new_key="taxonomy"),
            PipelineStep(name="conversation", old_key="conversation", new_key="conversation"),
            PipelineStep(name="length", old_key="length", new_key="length"),
        ]
        return pipeline_steps


class DPOPipelineBuilder:
    """Builds the pipeline steps for 'dpo' transformation."""

    def __init__(self):
        self.pipeline_steps = []

    def build_pipeline(self) -> list[PipelineStep]:
        """Build and return the pipeline for 'dpo' transformation."""
        pipeline_steps = [
            PipelineStep(name="copy", old_key="category", new_key="categories"),
            PipelineStep(name="copy", old_key="subcategory", new_key="subcategories"),
            PipelineStep(name="taxonomy", old_key="taxonomy", new_key="taxonomy"),
            PipelineStep(
                name="dpo_conversation",
                old_key="conversation",
                new_key="dpo_conversation",
            ),
            PipelineStep(name="length", old_key="length", new_key="length"),
        ]
        return pipeline_steps


class PipelineFactory:
    """Factory to create the appropriate pipeline based on transformation type."""

    def __init__(self, transform_type: str):
        """Initialize the factory with the transformation type (sft or dpo)."""
        self.transform_type = transform_type

    def get_pipeline(self) -> list[PipelineStep]:
        """Return the appropriate pipeline based on the transform type."""
        if self.transform_type == "sft":
            return SFTPipelineBuilder().build_pipeline()
        elif self.transform_type == "dpo":
            return DPOPipelineBuilder().build_pipeline()
        else:
            raise ValueError("Invalid transform type. Must be 'sft' or 'dpo'.")
