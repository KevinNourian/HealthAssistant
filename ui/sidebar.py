"""Sidebar UI — reminders, documents, settings, token usage."""

import logging
import os
from typing import Any

import streamlit as st

from core.user_data import save_user_data
from core.utils import calculate_session_cost

logger = logging.getLogger(__name__)


def _render_config_item(label: str, value: Any) -> None:
    """Render a single configuration value in the sidebar.

    Args:
        label: Short human-readable label.
        value: The configuration value to display.
    """
    st.markdown(
        f"""
        <div class="config-item">
            <div class="config-label">{label}</div>
            <div class="config-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render(
    username: str,
    name: str,
    authenticator: Any,
    config: dict[str, Any],
) -> None:
    """Render the sidebar.

    Args:
        username: The authenticated username.
        name: The user's display name.
        authenticator: The authenticator instance.
        config: The application configuration dictionary.
    """
    with st.sidebar:

        st.markdown(f"**Welcome, {name}!**")

        try:
            authenticator.logout(location="sidebar", button_name="Sign Out")
        except TypeError:
            authenticator.logout("Sign Out", "sidebar")

        st.markdown("---")

        # ── Reminders ────────────────────────────────────────────────────
        st.markdown("### ⏰ Reminders")

        with st.form("reminder_form", clear_on_submit=True):
            reminder_text = st.text_input(
                "Reminder",
                placeholder="e.g., Doctor visit",
                label_visibility="collapsed",
            )
            reminder_date = st.date_input("Date", label_visibility="collapsed")

            if st.form_submit_button("Add Reminder", use_container_width=True):
                if reminder_text:
                    st.session_state.reminders.append({
                        "text": reminder_text,
                        "date": reminder_date.strftime("%Y-%m-%d"),
                        "id": len(st.session_state.reminders),
                    })
                    save_user_data(username)
                    st.rerun()

        if st.session_state.reminders:
            for i, reminder in enumerate(st.session_state.reminders):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(
                        f"""
                        <div style="padding: 8px 0;
                                    border-bottom: 1px solid #E8E8E8;">
                            <span style="color: #6B7280; font-size: 12px;
                                   font-weight: 500;">{reminder['date']}</span>
                            <br>
                            <span style="color: #1F2937;
                                   font-size: 13px;">{reminder['text']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button(
                        "✕", key=f"del_rem_{i}", help="Delete",
                        use_container_width=True,
                    ):
                        st.session_state.reminders.pop(i)
                        save_user_data(username)
                        st.rerun()
        else:
            st.caption("No reminders yet")

        st.markdown("---")

        # ── Knowledge Base ───────────────────────────────────────────────
        st.markdown("### 📚 Documents")

        if config["pdf_files"]:
            for pdf in config["pdf_files"]:
                st.caption(f"• {os.path.basename(pdf)}")
        else:
            st.caption("No documents loaded")

        st.markdown("---")

        # ── Settings ─────────────────────────────────────────────────────
        st.markdown("### ⚙️ Settings")

        col1, col2 = st.columns(2)
        with col1:
            _render_config_item("Model", config["llm"]["model"])
            _render_config_item("Chunks", config["chunking"]["chunk_size"])
        with col2:
            _render_config_item("Temp", config["llm"]["temperature"])
            _render_config_item("Top-K", config["retriever"]["k"])

        st.markdown("---")

        # ── Token Usage ──────────────────────────────────────────────────
        st.markdown("### 📊 Token Usage")

        cost = calculate_session_cost(
            st.session_state.get("conversation_history", []),
            config["llm"]["model"],
        )

        _render_config_item("Input", f"{cost.total_input:,}")
        _render_config_item("Output", f"{cost.total_output:,}")
        _render_config_item("Cost", f"${cost.total_cost:.6f}")

        st.markdown("---")
        st.caption("Powered by OpenAI & LangChain")
