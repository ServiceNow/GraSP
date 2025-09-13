from pydantic import BaseModel, Field


class SimpleResponse(BaseModel):
    """Simple response with just text and status"""

    message: str = Field(description="Response message")
    success: bool = Field(default=True, description="Operation success status")
