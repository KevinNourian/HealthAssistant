"""
User data management and session state initialization.
"""

import json
import os

import streamlit as st


def load_user_data(username: str) -> dict:
    """Load user-specific data from JSON file."""
    filepath = f"user_data_{username}.json"
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {"reminders": [], "journal_entries": []}


def save_user_data(username: str) -> None:
    """Save user-specific data to JSON file."""
    filepath = f"user_data_{username}.json"
    data = {
        "reminders": st.session_state.reminders,
        "journal_entries": st.session_state.journal_entries,
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def init_session_state(username: str) -> None:
    """Initialize all session state variables for the given user.

    Reloads user data from disk when the active user changes and
    ensures every required session key exists with a sensible default.
    """
    # Reload from disk when the user changes
    if ("current_user" not in st.session_state
            or st.session_state.current_user != username):
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

    # Ensure every key exists (covers first-run & edge cases)
    defaults = {
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
