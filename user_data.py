"""
User data management and session state initialization.

Each user's reminders and journal entries are persisted as a JSON file
named ``user_data_<username>.json``.  Session state is re-initialized
whenever the active user changes (e.g. after a new login).
"""

import json
import logging
import os
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


def _user_data_path(username: str) -> str:
    """Return the JSON file path for the given user.

    Args:
        username: The authenticated username.

    Returns:
        A relative file path string.
    """
    return f"user_data_{username}.json"


def load_user_data(username: str) -> dict[str, Any]:
    """Load user-specific data from the JSON file on disk.

    If the file does not exist a default empty structure is returned
    so that the rest of the application can proceed normally.

    Args:
        username: The authenticated username.

    Returns:
        A dictionary with ``reminders`` and ``journal_entries`` keys.
    """
    filepath = _user_data_path(username)
    if not os.path.exists(filepath):
        logger.info("No data file for user '%s'; returning defaults", username)
        return {"reminders": [], "journal_entries": []}

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        logger.info("Loaded data for user '%s'", username)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load data for user '%s': %s", username, e)
        return {"reminders": [], "journal_entries": []}


def save_user_data(username: str) -> None:
    """Persist the current session-state data for the given user.

    Reads ``st.session_state.reminders`` and
    ``st.session_state.journal_entries`` and writes them to disk.

    Args:
        username: The authenticated username.
    """
    filepath = _user_data_path(username)
    data: dict[str, Any] = {
        "reminders": st.session_state.reminders,
        "journal_entries": st.session_state.journal_entries,
    }
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved data for user '%s'", username)
    except OSError as e:
        logger.error("Failed to save data for user '%s': %s", username, e)


def init_session_state(username: str) -> None:
    """Initialize all session-state variables for the given user.

    Reloads user data from disk when the active user changes and
    ensures every required session key exists with a sensible default.
    This is safe to call on every Streamlit rerun.

    Args:
        username: The authenticated username.
    """
    # Reload from disk when the user changes
    if (
        "current_user" not in st.session_state
        or st.session_state.current_user != username
    ):
        logger.info("Initializing session state for user '%s'", username)
        user_data = load_user_data(username)
        st.session_state.reminders = user_data.get("reminders", [])
        st.session_state.journal_entries = user_data.get("journal_entries", [])
        st.session_state.current_user = username
        st.session_state.file_uploader_key = 0
        st.session_state.journal_form_key = 0
        st.session_state.editing_entry = None
        st.session_state.conversation_history = []
        st.session_state.agent_messages = []
        st.session_state.question_counter = 0

    # Ensure every key exists (covers first-run and edge cases)
    defaults: dict[str, Any] = {
        "file_uploader_key": 0,
        "journal_form_key": 0,
        "editing_entry": None,
        "conversation_history": [],
        "agent_messages": [],
        "question_counter": 0,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
