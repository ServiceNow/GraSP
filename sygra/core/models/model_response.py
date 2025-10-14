from typing import Optional

from pydantic import BaseModel


class ModelResponse(BaseModel):
    """
    Model response object
    """

    llm_response: str
    response_code: int
    reasoning_response: Optional[str] = None
    finish_reason: Optional[str] = None
    tool_calls: Optional[list] = None
