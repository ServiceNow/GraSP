import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
import requests  # type: ignore[import-untyped]
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field

from grasp.core.models.client.base_client import BaseClient
from grasp.logger.logger_config import logger
from grasp.utils import constants


class HttpClientConfig(BaseModel):
    """Configuration model for the HTTP client"""

    base_url: str = Field(..., description="Base URL for the API")
    headers: Dict[str, str] = Field(
        default_factory=dict, description="Headers to include in all requests"
    )
    timeout: int = Field(
        default=constants.DEFAULT_TIMEOUT, description="Request timeout in seconds"
    )
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")
    ssl_verify: bool = Field(default=True, description="Verify SSL certificate")
    ssl_cert: Optional[str] = Field(default=None, description="Path to SSL certificate file")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )


class HttpClient(BaseClient):
    """
    Generic HTTP client for making API calls.

    This client provides a standardized interface for building HTTP requests
    for various API endpoints, designed to be compatible with CustomTGI and
    CustomAzure classes in the GraSP framework.
    """

    def __init__(self, stop: Optional[List[str]] = None, **client_kwargs):
        """
        Initialize an HTTP client.

        Args:
            stop (Optional[List[str]], optional): List of strings to stop generation at. Defaults to None.
            **client_kwargs: Additional keyword arguments for client configuration.
        """
        super().__init__(**client_kwargs)

        # Validate configuration using Pydantic model
        validated_config = HttpClientConfig(**client_kwargs)

        self.base_url = validated_config.base_url
        self.headers = validated_config.headers
        self.timeout = validated_config.timeout
        self.max_retries = validated_config.max_retries
        self.verify_ssl = validated_config.ssl_verify
        self.verify_cert = validated_config.ssl_cert
        self.stop = stop

    def build_request(
        self,
        messages: Optional[Sequence[BaseMessage]] = None,
        formatted_prompt: Optional[str] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        # build_request is not supported in this class
        # Use build_request_with_payload instead to construct the payload
        raise NotImplementedError(
            "HttpClient.build_request is not supported. Use build_request_with_payload(payload, **kwargs) instead."
        )

    def build_request_with_payload(self, payload: Dict[str, Any], **kwargs):
        """
        Build a request payload for the API.

        Args:
            payload (Dict[str, Any]): The payload to include in the request.
            **kwargs: Additional keyword arguments to include in the payload.

        Returns:
            dict: The request payload.

        Raises:
            ValueError: If required parameters are missing based on the API type.
        """
        # Include stop sequences if specified
        if self.stop is not None:
            kwargs["stop"] = self.stop

        payload.update(kwargs)
        return payload

    def send_request(
        self,
        payload: Dict[str, Any],
        model_name: Optional[str] = None,
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Send an HTTP request to the API endpoint.

        This method sends the actual request using aiohttp and returns the response text and status.

        Args:
            payload (Dict[str, Any]): The payload to send to the API.
            model_name (str, optional): Model name to use in the request. Defaults to None.
            generation_params (Optional[Dict[str, Any]], optional): Additional generation parameters. Defaults to None.

        Returns:
            ClientResponse: The response from the API.
        """
        # Update payload with generation parameters if provided
        if generation_params:
            payload.update(generation_params)

        try:
            # Convert payload to JSON string
            json_data = json.dumps(payload).encode()

            response = requests.request(
                "POST",
                self.base_url,
                headers=self.headers,
                data=json_data,
                timeout=self.timeout,
                verify=self.verify_ssl,
                cert=self.verify_cert,
            )

        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return ""
        return response

    async def async_send_request(
        self,
        payload: Dict[str, Any],
        model_name: Optional[str] = None,
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Send an HTTP request to the API endpoint.

        This method sends the actual request using aiohttp and returns the response text and status.

        Args:
            payload (Dict[str, Any]): The payload to send to the API.
            model_name (str, optional): Model name to use in the request. Defaults to None.
            generation_params (Optional[Dict[str, Any]], optional): Additional generation parameters. Defaults to None.

        Returns:
            ClientResponse: The response from the API.
        """
        # Update payload with generation parameters if provided
        if generation_params:
            payload.update(generation_params)

        try:
            # Convert payload to JSON string
            json_data = json.dumps(payload).encode()

            # Send request using aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    data=json_data,
                    headers=self.headers,
                    timeout=self.timeout,
                    ssl=self.verify_ssl,
                ) as resp:
                    # Read the body text to ensure the content is consumed before returning
                    body_text = await resp.text()
                    # Return a lightweight object mirroring requests.Response essentials
                    return SimpleNamespace(
                        text=body_text,
                        status=resp.status,
                        status_code=resp.status,
                        headers=dict(resp.headers),
                    )
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return ""
