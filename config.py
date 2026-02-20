"""
Configuration loading and application constants.
"""

import json


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
SUMMARY_MAX_CHARS = 3000
LAB_REPORT_MAX_CHARS = 4000


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG FILE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config_data: dict, config_path: str = "config.json") -> None:
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
