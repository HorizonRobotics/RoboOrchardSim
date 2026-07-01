# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Standalone GPT client for vision-language queries.

Supports both Azure OpenAI and standard OpenAI API endpoints.
"""

import base64
import logging
import os
from io import BytesIO
from typing import Optional

import yaml
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def _openai_module():
    try:
        import openai
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "The asset pipeline GPT client requires the optional "
            "`openai` package. Install tools/asset_pipeline/requirements.txt "
            "before constructing GPTClient."
        ) from e
    return openai


def _is_retryable_openai_error(exc: BaseException) -> bool:
    try:
        openai = _openai_module()
    except ModuleNotFoundError:
        return False
    return not isinstance(exc, openai.BadRequestError)


class GPTClient:
    """A client for querying GPT models with text and image inputs.

    Supports Azure OpenAI and standard OpenAI API backends with
    automatic retry and backoff.

    Args:
        endpoint: API endpoint URL.
        api_key: API key for authentication.
        model_name: Model name to use (e.g., "gpt-4o").
        api_version: API version string (Azure only, None for OpenAI).
        check_connection: Whether to verify API connection on init.
        verbose: Enable verbose logging of prompts and responses.

    Example:
        ```python
        client = GPTClient(
            endpoint="https://api.openai.com/v1",
            api_key="sk-xxx",
            model_name="gpt-4o",
        )
        response = client.query(
            "Describe this object.",
            images=["path/to/image.png"],
        )
        ```
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model_name: str = "gpt-4o",
        api_version: str = None,
        check_connection: bool = True,
        verbose: bool = False,
    ):
        openai = _openai_module()
        if api_version is not None:
            self.client = openai.AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        else:
            self.client = openai.OpenAI(
                base_url=endpoint,
                api_key=api_key,
            )

        self.endpoint = endpoint
        self.model_name = model_name
        self.verbose = verbose

        if check_connection:
            self.check_connection()

        logger.info(f"GPTClient initialized with model: {self.model_name}")

    @retry(
        retry=retry_if_exception(_is_retryable_openai_error),
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(5),
    )
    def _completion_with_backoff(self, **kwargs):
        """Perform a chat completion request with retry and backoff."""
        return self.client.chat.completions.create(**kwargs)

    def query(
        self,
        text_prompt: str,
        images: Optional[list[str | Image.Image]] = None,
        system_role: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> Optional[str]:
        """Query the GPT model with text and optional images.

        Args:
            text_prompt: The text prompt to send.
            images: Optional list of image file paths or PIL Images.
            system_role: System-level instructions for the model.
            params: Additional parameters to override defaults
                (e.g., temperature, max_tokens).

        Returns:
            Model response text, or None if an error occurred.
        """
        if system_role is None:
            system_role = (
                "You are a highly knowledgeable assistant specializing in "
                "physics, engineering, and object properties."
            )

        # Build user content
        content_user = [{"type": "text", "text": text_prompt}]

        if images is not None:
            if not isinstance(images, list):
                images = [images]

            for img in images:
                img_b64 = encode_image_to_base64(img)
                content_user.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        },
                    }
                )

        payload = {
            "messages": [
                {"role": "system", "content": system_role},
                {"role": "user", "content": content_user},
            ],
            "temperature": 0.1,
            "max_completion_tokens": 2000,
            "top_p": 0.1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "model": self.model_name,
        }

        if params:
            payload.update(params)

        response = None
        try:
            response = self._completion_with_backoff(**payload)
            response = response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT API call failed ({self.endpoint}): {e}")
            response = None

        if self.verbose:
            logger.info(f"Prompt: {text_prompt}")
            logger.info(f"Response: {response}")

        return response

    def check_connection(self) -> None:
        """Verify that the GPT API connection is working.

        Raises:
            ConnectionError: If the connection check fails.
        """
        try:
            response = self._completion_with_backoff(
                messages=[
                    {"role": "system", "content": "You are a test system."},
                    {"role": "user", "content": "Hello"},
                ],
                model=self.model_name,
                temperature=0,
                max_completion_tokens=100,
            )
            _ = response.choices[0].message.content
            logger.info("GPT connection check passed.")
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to GPT API at {self.endpoint}: {e}"
            )


def encode_image_to_base64(image: str | Image.Image) -> str:
    """Encode an image to a base64 string.

    Args:
        image: A file path string or PIL Image.

    Returns:
        Base64-encoded string of the image.

    Raises:
        FileNotFoundError: If the image file path does not exist.
        ValueError: If the input type is not recognized.
    """
    if isinstance(image, Image.Image):
        buffer = BytesIO()
        fmt = image.format or "PNG"
        image.save(buffer, format=fmt)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    if isinstance(image, str):
        ext = os.path.splitext(image)[-1].lower()
        if ext in IMAGE_FORMATS:
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            # Assume it's already a base64 string
            return image

    raise ValueError(f"Unsupported image type: {type(image)}")


def load_client_from_config(
    config_path: str, check_connection: bool = False
) -> GPTClient:
    """Create a GPTClient from a YAML configuration file.

    The config file should have the structure:
        agent_type: "gpt-4o"
        gpt-4o:
            endpoint: https://...
            api_key: xxx
            api_version: xxx  # optional, for Azure
            model_name: gpt-4o

    Environment variables ENDPOINT, API_KEY, API_VERSION, MODEL_NAME
    take precedence over config values.

    Args:
        config_path: Path to the YAML config file.
        check_connection: Whether to verify API connection.

    Returns:
        A configured GPTClient instance.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    agent_type = config["agent_type"]
    agent_config = config.get(agent_type, {})

    endpoint = os.environ.get("ENDPOINT", agent_config.get("endpoint"))
    api_key = os.environ.get("API_KEY", agent_config.get("api_key"))
    api_version = os.environ.get(
        "API_VERSION", agent_config.get("api_version")
    )
    model_name = os.environ.get("MODEL_NAME", agent_config.get("model_name"))

    return GPTClient(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        model_name=model_name,
        check_connection=check_connection,
    )
