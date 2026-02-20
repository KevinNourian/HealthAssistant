"""
Configuration loading and application constants.

Provides helpers to read and write the JSON configuration file that
controls LLM parameters, chunking settings, retriever options, and
the list of PDF documents in the knowledge base.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
SUMMARY_MAX_CHARS: int = 3000
"""Maximum characters of document text sent to the summary prompt."""

LAB_REPORT_MAX_CHARS: int = 4000
"""Maximum characters of lab report text sent to the analysis prompt."""


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
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info("Configuration loaded successfully")
        return config
    except FileNotFoundError:
        logger.error("Configuration file not found: %s", config_path)
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in configuration file: %s", e)
        raise


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
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info("Configuration saved successfully")
    except OSError as e:
        logger.error("Failed to save configuration: %s", e)
        raise
