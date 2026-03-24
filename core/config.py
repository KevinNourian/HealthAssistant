"""
Configuration loading and application constants.

Provides helpers to read and write the JSON configuration file that
controls LLM parameters, chunking settings, retriever options, and
the list of PDF documents in the knowledge base.
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PRICING (USD per 1M tokens)
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG FILE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def load_config(config_path: str = "config.json") -> dict[str, Any]:
    """Load application configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        A dictionary containing all configuration values.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    logger.info("Loading configuration from %s", config_path)
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        logger.info("Configuration loaded successfully")
        return config
    except FileNotFoundError:
        logger.error("Configuration file not found: %s", config_path)
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in configuration file: %s", e)
        raise


def validate_api_keys() -> None:
    """Validate that required API keys are present in the environment.

    Reads ``OPENAI_API_KEY`` and ``SERPAPI_API_KEY`` from environment
    variables.

    Raises:
        EnvironmentError: If ``OPENAI_API_KEY`` is missing or does not
            look like a valid OpenAI key (``sk-...``).
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file and restart the app."
        )
    if not openai_key.startswith("sk-"):
        raise EnvironmentError(
            "OPENAI_API_KEY appears invalid — expected a key starting with "
            "'sk-'. Check your .env file."
        )

    serpapi_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not serpapi_key:
        logger.warning(
            "SERPAPI_API_KEY is not set — the web search tool will be "
            "unavailable"
        )
    else:
        logger.info("API key validation passed")


def save_config(
    config_data: dict[str, Any],
    config_path: str = "config.json",
) -> None:
    """Save application configuration to a JSON file.

    Args:
        config_data: The configuration dictionary to persist.
        config_path: Path to the JSON configuration file.

    Raises:
        OSError: If the file cannot be written.
    """
    logger.info("Saving configuration to %s", config_path)
    try:
        with open(config_path, 'w') as config_file:
            json.dump(config_data, config_file, indent=2)
        logger.info("Configuration saved successfully")
    except OSError as e:
        logger.error("Failed to save configuration: %s", e)
        raise
