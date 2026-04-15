"""Vaccination tab UI — vaccination tracking."""

import logging
from datetime import datetime
from typing import Any

import streamlit as st

from core.user_data import save_user_data

logger = logging.getLogger(__name__)


def render(username: str) -> None:
    """Render the Vaccination tab.

    Args:
        username: The authenticated username.
    """
    st.markdown("### Vaccination")
    st.caption("Record and track your vaccinations")

    # ── Entry Form ───────────────────────────────────────────────────────
    vax_key = st.session_state.vax_form_key
    vax_cols = st.columns([1, 2])

    with vax_cols[0]:
        vax_date = st.date_input("Date", key=f"vax_date_{vax_key}")
    with vax_cols[1]:
        vax_name = st.text_input(
            "Name of Vaccination", key=f"vax_name_{vax_key}",
        )

    st.caption(
        "💡 Formatting: `**bold**`, `*italic*`, "
        "`- bullet point`, `1. numbered list`"
    )
    vax_notes = st.text_area(
        "Notes",
        key=f"vax_notes_{vax_key}",
        height=120,
    )

    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Save Vaccination", use_container_width=True):
            if not vax_name.strip():
                st.warning("Please enter the name of the vaccination.")
            else:
                vaccination: dict[str, Any] = {
                    "date": vax_date.strftime("%Y-%m-%d"),
                    "name": vax_name.strip(),
                    "notes": vax_notes.strip(),
                }
                st.session_state.vaccinations.append(vaccination)
                save_user_data(username)
                st.session_state.vax_form_key += 1
                logger.info("Vaccination saved: %s", vaccination)
                st.rerun()
    with col_cancel:
        if st.button(
            "Cancel", key="vax_cancel", use_container_width=True,
        ):
            st.session_state.vax_form_key += 1
            st.rerun()

    # ── Records ──────────────────────────────────────────────────────────
    if st.session_state.vaccinations:
        st.markdown("---")
        st.markdown("#### Records")

        for idx, vax in enumerate(
            reversed(st.session_state.vaccinations)
        ):
            real_idx: int = (
                len(st.session_state.vaccinations) - 1 - idx
            )
            is_editing_vax: bool = (
                st.session_state.editing_vaccination == real_idx
            )

            with st.container(border=True):
                if is_editing_vax:
                    st.markdown("**Editing Vaccination**")

                    edit_vax_cols = st.columns([1, 2])
                    with edit_vax_cols[0]:
                        edited_vax_date = st.date_input(
                            "Date",
                            value=datetime.strptime(
                                vax["date"], "%Y-%m-%d"
                            ).date(),
                            key=f"edit_vax_date_{real_idx}",
                        )
                    with edit_vax_cols[1]:
                        edited_vax_name = st.text_input(
                            "Name of Vaccination",
                            value=vax["name"],
                            key=f"edit_vax_name_{real_idx}",
                        )

                    st.caption(
                        "💡 Formatting: `**bold**`, `*italic*`, "
                        "`- bullet point`, `1. numbered list`"
                    )
                    edited_vax_notes = st.text_area(
                        "Notes",
                        value=vax.get("notes", ""),
                        height=120,
                        key=f"edit_vax_notes_{real_idx}",
                    )

                    col_save_edit, col_cancel_edit = st.columns(2)
                    with col_save_edit:
                        if st.button(
                            "Save Changes",
                            key=f"save_vax_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.vaccinations[real_idx] = {
                                "date": edited_vax_date.strftime("%Y-%m-%d"),
                                "name": edited_vax_name.strip(),
                                "notes": edited_vax_notes.strip(),
                            }
                            save_user_data(username)
                            st.session_state.editing_vaccination = None
                            logger.info("Vaccination updated")
                            st.rerun()
                    with col_cancel_edit:
                        if st.button(
                            "Cancel",
                            key=f"cancel_vax_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_vaccination = None
                            st.rerun()
                else:
                    col_date, col_name, col_del = st.columns(
                        [2, 3, 1]
                    )

                    with col_date:
                        parsed_date = datetime.strptime(
                            vax["date"], "%Y-%m-%d"
                        )
                        display_date = (
                            f"{parsed_date.strftime('%B')} "
                            f"{parsed_date.day}, "
                            f"{parsed_date.year}"
                        )
                        st.markdown(f"**{display_date}**")
                    with col_name:
                        st.markdown(f"**{vax['name']}**")
                    with col_del:
                        st.markdown("")

                    if vax.get("notes"):
                        st.markdown("---")
                        st.markdown(vax["notes"])

                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(
                            "Edit",
                            key=f"edit_vax_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_vaccination = real_idx
                            st.rerun()
                    with col_delete:
                        if st.button(
                            "Delete",
                            key=f"del_vax_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.vaccinations.pop(real_idx)
                            save_user_data(username)
                            logger.info("Vaccination deleted")
                            st.rerun()
    else:
        st.info(
            "No vaccinations yet. Start tracking your vaccinations!"
        )
