"""Health Journal tab UI — journaling with attachments."""

import logging
import os
from datetime import datetime
from typing import Any

import streamlit as st

from core.user_data import save_user_data

logger = logging.getLogger(__name__)


def render(username: str) -> None:
    """Render the Health Journal tab.

    Args:
        username: The authenticated username.
    """
    st.markdown("### Health Journal")
    st.caption("Track your health journey with notes and attachments")

    st.caption(
        "💡 Formatting: `**bold**`, `*italic*`, "
        "`- bullet point`, `1. numbered list`"
    )

    col_title, col_date = st.columns([3, 1])

    with col_title:
        journal_title: str = st.text_input(
            "Title",
            placeholder="Entry title...",
            key=f"journal_title_{st.session_state.journal_form_key}",
        )

    with col_date:
        journal_date = st.date_input("Date", key="journal_date")

    journal_entry: str = st.text_area(
        "Entry",
        placeholder="How are you feeling today?",
        height=120,
        key=f"journal_entry_{st.session_state.journal_form_key}",
    )

    uploaded_file = st.file_uploader(
        "Attachment (optional)",
        type=["pdf", "png", "jpg", "jpeg", "gif"],
        key=f"journal_file_{st.session_state.file_uploader_key}",
    )

    if st.button("Save Entry", use_container_width=True):
        if journal_title and journal_entry:
            entry_data: dict[str, Any] = {
                "title": journal_title,
                "date": journal_date.strftime("%Y-%m-%d"),
                "entry": journal_entry,
                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }

            if uploaded_file is not None:
                attachment_dir: str = (
                    f"journal_attachments/{username}"
                )
                os.makedirs(attachment_dir, exist_ok=True)

                timestamp: str = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
                file_extension: str = (
                    uploaded_file.name.rsplit(".", 1)[-1].lower()
                )
                safe_filename: str = (
                    f"{timestamp}_{uploaded_file.name}"
                )
                file_path: str = os.path.join(
                    attachment_dir, safe_filename
                )

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                entry_data["attachment"] = {
                    "filename": uploaded_file.name,
                    "filepath": file_path,
                    "type": file_extension,
                }
                logger.info(
                    "Journal attachment saved: %s", file_path,
                )

            st.session_state.journal_entries.append(entry_data)
            save_user_data(username)
            st.session_state.file_uploader_key += 1
            st.session_state.journal_form_key += 1
            logger.info("Journal entry saved: %s", journal_title)
            st.rerun()
        else:
            st.warning("Please enter both title and entry")

    st.markdown("---")

    # Past Entries
    if st.session_state.journal_entries:
        st.markdown("#### Past Entries")

        for entry in reversed(st.session_state.journal_entries):
            attachment_icon: str = " 📎" if "attachment" in entry else ""
            is_editing: bool = (
                st.session_state.editing_entry == entry["timestamp"]
            )

            with st.container(border=True):
                if is_editing:
                    st.markdown("**✏️ Editing Entry**")

                    edited_title: str = st.text_input(
                        "Title",
                        value=entry.get("title", ""),
                        key=f"edit_title_{entry['timestamp']}",
                    )

                    edited_entry_text: str = st.text_area(
                        "Entry",
                        value=entry["entry"],
                        height=120,
                        key=f"edit_entry_{entry['timestamp']}",
                    )

                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button(
                            "Save Changes",
                            key=f"save_{entry['timestamp']}",
                            use_container_width=True,
                        ):
                            for idx, e in enumerate(
                                st.session_state.journal_entries
                            ):
                                if e["timestamp"] == entry["timestamp"]:
                                    st.session_state.journal_entries[idx][
                                        "title"
                                    ] = edited_title
                                    st.session_state.journal_entries[idx][
                                        "entry"
                                    ] = edited_entry_text
                                    break
                            save_user_data(username)
                            st.session_state.editing_entry = None
                            logger.info("Journal entry updated")
                            st.success("Entry updated!")
                            st.rerun()

                    with col_cancel:
                        if st.button(
                            "Cancel",
                            key=f"cancel_{entry['timestamp']}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_entry = None
                            st.rerun()
                else:
                    parsed_journal_date = datetime.strptime(
                        entry["date"], "%Y-%m-%d"
                    )
                    journal_display_date = (
                        f"{parsed_journal_date.strftime('%B')} "
                        f"{parsed_journal_date.day}, "
                        f"{parsed_journal_date.year}"
                    )
                    st.markdown(
                        f"**{journal_display_date} — "
                        f"{entry.get('title', 'Untitled')}"
                        f"{attachment_icon}**"
                    )
                    st.markdown(entry["entry"])

                    if "attachment" in entry:
                        st.markdown("---")
                        attachment: dict[str, str] = entry["attachment"]
                        file_type: str = attachment["type"]

                        if file_type in ["png", "jpg", "jpeg", "gif"]:
                            st.image(
                                attachment["filepath"],
                                caption=attachment["filename"],
                                use_container_width=True,
                            )
                        elif file_type == "pdf":
                            st.caption(f"📄 {attachment['filename']}")
                            with open(attachment["filepath"], "rb") as f:
                                st.download_button(
                                    label="Download PDF",
                                    data=f,
                                    file_name=attachment["filename"],
                                    mime="application/pdf",
                                    key=f"dl_{entry['timestamp']}",
                                )

                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(
                            "Edit Entry",
                            key=f"edit_{entry['timestamp']}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_entry = (
                                entry["timestamp"]
                            )
                            st.rerun()

                    with col_delete:
                        if st.button(
                            "Delete Entry",
                            key=f"del_{entry['timestamp']}",
                            use_container_width=True,
                        ):
                            if (
                                "attachment" in entry
                                and "filepath" in entry["attachment"]
                            ):
                                filepath: str = os.path.normpath(
                                    entry["attachment"]["filepath"]
                                )
                                if os.path.exists(filepath):
                                    try:
                                        os.remove(filepath)
                                    except OSError as e:
                                        logger.error(
                                            "Failed to delete attachment "
                                            "%s: %s", filepath, e,
                                        )

                            for idx, e in enumerate(
                                st.session_state.journal_entries
                            ):
                                if e["timestamp"] == entry["timestamp"]:
                                    st.session_state.journal_entries.pop(idx)
                                    break

                            save_user_data(username)
                            logger.info("Journal entry deleted")
                            st.rerun()
    else:
        st.info(
            "No journal entries yet. Start tracking your health journey!"
        )
