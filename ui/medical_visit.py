"""Medical Visit tab UI — appointment tracking."""

import logging
from datetime import datetime
from typing import Any

import streamlit as st

from core.user_data import save_user_data

logger = logging.getLogger(__name__)


def render(username: str) -> None:
    """Render the Medical Visit tab.

    Args:
        username: The authenticated username.
    """
    st.markdown("### Medical Visit")
    st.caption("Record and track your medical appointments")

    # ── Entry Form ───────────────────────────────────────────────────────
    mv_key = st.session_state.mv_form_key
    mv_cols = st.columns([1, 1, 2])

    with mv_cols[0]:
        mv_date = st.date_input("Date", key=f"mv_date_{mv_key}")
    with mv_cols[1]:
        mv_time = st.time_input("Time", key=f"mv_time_{mv_key}")
    with mv_cols[2]:
        mv_doctor = st.text_input("Doctor", key=f"mv_doctor_{mv_key}")

    st.caption(
        "💡 Formatting: `**bold**`, `*italic*`, "
        "`- bullet point`, `1. numbered list`"
    )
    mv_notes = st.text_area(
        "Notes",
        key=f"mv_notes_{mv_key}",
        height=120,
    )

    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Save Visit", use_container_width=True):
            if not mv_doctor.strip():
                st.warning("Please enter a doctor name.")
            else:
                visit: dict[str, Any] = {
                    "date": mv_date.strftime("%Y-%m-%d"),
                    "time": mv_time.strftime("%H:%M"),
                    "doctor": mv_doctor.strip(),
                    "notes": mv_notes.strip(),
                }
                st.session_state.medical_visits.append(visit)
                save_user_data(username)
                st.session_state.mv_form_key += 1
                logger.info("Medical visit saved: %s", visit)
                st.rerun()
    with col_cancel:
        if st.button("Cancel", key="mv_cancel", use_container_width=True):
            st.session_state.mv_form_key += 1
            st.rerun()

    # ── Visits ────────────────────────────────────────────────────────────
    if st.session_state.medical_visits:
        st.markdown("---")
        st.markdown("#### Visits")

        for idx, visit in enumerate(
            reversed(st.session_state.medical_visits)
        ):
            real_idx: int = (
                len(st.session_state.medical_visits) - 1 - idx
            )
            is_editing_visit: bool = (
                st.session_state.editing_visit == real_idx
            )

            with st.container(border=True):
                if is_editing_visit:
                    st.markdown("**Editing Visit**")

                    edit_mv_cols = st.columns([1, 1, 2])
                    with edit_mv_cols[0]:
                        edited_mv_date = st.date_input(
                            "Date",
                            value=datetime.strptime(
                                visit["date"], "%Y-%m-%d"
                            ).date(),
                            key=f"edit_mv_date_{real_idx}",
                        )
                    with edit_mv_cols[1]:
                        edited_mv_time = st.time_input(
                            "Time",
                            value=datetime.strptime(
                                visit["time"], "%H:%M"
                            ).time(),
                            key=f"edit_mv_time_{real_idx}",
                        )
                    with edit_mv_cols[2]:
                        edited_mv_doctor = st.text_input(
                            "Doctor",
                            value=visit["doctor"],
                            key=f"edit_mv_doctor_{real_idx}",
                        )

                    st.caption(
                        "💡 Formatting: `**bold**`, `*italic*`, "
                        "`- bullet point`, `1. numbered list`"
                    )
                    edited_mv_notes = st.text_area(
                        "Notes",
                        value=visit.get("notes", ""),
                        height=120,
                        key=f"edit_mv_notes_{real_idx}",
                    )

                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button(
                            "Save Changes",
                            key=f"save_mv_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.medical_visits[real_idx] = {
                                "date": edited_mv_date.strftime("%Y-%m-%d"),
                                "time": edited_mv_time.strftime("%H:%M"),
                                "doctor": edited_mv_doctor.strip(),
                                "notes": edited_mv_notes.strip(),
                            }
                            save_user_data(username)
                            st.session_state.editing_visit = None
                            logger.info("Medical visit updated")
                            st.rerun()
                    with col_cancel:
                        if st.button(
                            "Cancel",
                            key=f"cancel_mv_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_visit = None
                            st.rerun()
                else:
                    col_date, col_doctor, col_del = st.columns(
                        [2, 3, 1]
                    )

                    with col_date:
                        parsed_date = datetime.strptime(
                            visit["date"], "%Y-%m-%d"
                        )
                        display_date = (
                            f"{parsed_date.strftime('%B')} "
                            f"{parsed_date.day}, "
                            f"{parsed_date.year}"
                        )
                        st.markdown(
                            f"**{display_date}**  \n"
                            f"{visit.get('time', '')}"
                        )
                    with col_doctor:
                        st.markdown(f"**Dr.** {visit['doctor']}")
                    with col_del:
                        st.markdown("")

                    if visit.get("notes"):
                        st.markdown("---")
                        st.markdown(visit["notes"])

                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(
                            "Edit",
                            key=f"edit_mv_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_visit = real_idx
                            st.rerun()
                    with col_delete:
                        if st.button(
                            "Delete",
                            key=f"del_mv_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.medical_visits.pop(real_idx)
                            save_user_data(username)
                            logger.info("Medical visit deleted")
                            st.rerun()
    else:
        st.info(
            "No visits yet. Start tracking your medical appointments!"
        )
