"""Lab Test tab UI — lab result tracking with PDF attachments."""

import logging
import os
from datetime import datetime
from typing import Any

import streamlit as st

from core.user_data import save_user_data

logger = logging.getLogger(__name__)


def render(username: str) -> None:
    """Render the Lab Test tab.

    Args:
        username: The authenticated username.
    """
    st.markdown("### Lab Test")
    st.caption("Record and track your lab test results")

    # ── Entry Form ───────────────────────────────────────────────────────
    lab_key = st.session_state.lab_form_key
    item_key = st.session_state.lab_item_key

    lab_date = st.date_input("Date", key=f"lab_date_{lab_key}")

    # ── Test / Result / Normal builder ───────────────────────────────────
    st.markdown("**Type**")

    # Display already-added test items
    for t_idx, test_item in enumerate(
        st.session_state.pending_lab_items
    ):
        is_editing_item: bool = (
            st.session_state.editing_lab_item
            == f"pending_{t_idx}"
        )
        with st.container(border=True):
            if is_editing_item:
                ei_cols = st.columns([2, 2, 2, 1, 1])
                with ei_cols[0]:
                    edited_test = st.text_input(
                        "Test",
                        value=test_item["test"],
                        key=f"ledit_test_{t_idx}_{lab_key}",
                        label_visibility="collapsed",
                    )
                with ei_cols[1]:
                    edited_result = st.text_input(
                        "Result",
                        value=test_item["result"],
                        key=f"ledit_result_{t_idx}_{lab_key}",
                        label_visibility="collapsed",
                    )
                with ei_cols[2]:
                    edited_normal = st.text_input(
                        "Normal",
                        value=test_item["normal"],
                        key=f"ledit_normal_{t_idx}_{lab_key}",
                        label_visibility="collapsed",
                    )
                with ei_cols[3]:
                    if st.button(
                        "Save",
                        key=f"ledit_save_{t_idx}_{lab_key}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_lab_items[t_idx] = {
                            "test": edited_test.strip(),
                            "result": edited_result.strip(),
                            "normal": edited_normal.strip(),
                        }
                        st.session_state.editing_lab_item = None
                        st.rerun()
                with ei_cols[4]:
                    if st.button(
                        "Cancel",
                        key=f"ledit_cancel_{t_idx}_{lab_key}",
                        use_container_width=True,
                    ):
                        st.session_state.editing_lab_item = None
                        st.rerun()
            else:
                t_col_test, t_col_sep1, t_col_result, t_col_sep2, t_col_normal, t_col_edit, t_col_trash = (
                    st.columns([2, 0.15, 2, 0.15, 2, 0.5, 0.5])
                )
                with t_col_test:
                    st.markdown(f"**{test_item['test']}**")
                with t_col_sep1:
                    st.markdown(
                        "<div style='border-left: 2px solid #ccc;"
                        " height: 100%; min-height: 30px;'></div>",
                        unsafe_allow_html=True,
                    )
                with t_col_result:
                    st.markdown(test_item["result"])
                with t_col_sep2:
                    st.markdown(
                        "<div style='border-left: 2px solid #ccc;"
                        " height: 100%; min-height: 30px;'></div>",
                        unsafe_allow_html=True,
                    )
                with t_col_normal:
                    st.markdown(test_item["normal"])
                with t_col_edit:
                    if st.button(
                        "\u270F\uFE0F",
                        key=f"pen_lab_{t_idx}_{lab_key}",
                    ):
                        st.session_state.editing_lab_item = (
                            f"pending_{t_idx}"
                        )
                        st.rerun()
                with t_col_trash:
                    if st.button(
                        "\U0001f5d1\uFE0F",
                        key=f"trash_lab_{t_idx}_{lab_key}",
                    ):
                        st.session_state.pending_lab_items.pop(t_idx)
                        st.rerun()

    # Input row for new test item
    add_cols = st.columns([2, 2, 2, 1])
    with add_cols[0]:
        new_test = st.text_input(
            "Test", key=f"lab_test_{item_key}",
            label_visibility="collapsed", placeholder="Test",
        )
    with add_cols[1]:
        new_result = st.text_input(
            "Result", key=f"lab_result_{item_key}",
            label_visibility="collapsed", placeholder="Result",
        )
    with add_cols[2]:
        new_normal = st.text_input(
            "Normal", key=f"lab_normal_{item_key}",
            label_visibility="collapsed", placeholder="Normal",
        )
    with add_cols[3]:
        if st.button("Add", key=f"lab_add_{item_key}",
                      use_container_width=True):
            if (
                new_test.strip()
                and new_result.strip()
                and new_normal.strip()
            ):
                st.session_state.pending_lab_items.append({
                    "test": new_test.strip(),
                    "result": new_result.strip(),
                    "normal": new_normal.strip(),
                })
                st.session_state.lab_item_key += 1
                st.rerun()
            else:
                st.warning("Please enter test, result, and normal range.")

    lab_doctor = st.text_input("Doctor", key=f"lab_doctor_{lab_key}")

    st.caption(
        "💡 Formatting: `**bold**`, `*italic*`, "
        "`- bullet point`, `1. numbered list`"
    )
    lab_notes = st.text_area(
        "Notes",
        key=f"lab_notes_{lab_key}",
        height=120,
    )

    lab_uploaded_file = st.file_uploader(
        "Attach PDF (optional)",
        type=["pdf"],
        key=f"lab_file_{st.session_state.lab_file_key}",
    )

    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Save Lab Test", use_container_width=True):
            if not st.session_state.pending_lab_items:
                st.warning("Please add at least one test result.")
            elif not lab_doctor.strip():
                st.warning("Please enter a doctor name.")
            else:
                lab_test: dict[str, Any] = {
                    "date": lab_date.strftime("%Y-%m-%d"),
                    "tests": list(st.session_state.pending_lab_items),
                    "doctor": lab_doctor.strip(),
                    "notes": lab_notes.strip(),
                }

                if lab_uploaded_file is not None:
                    lab_attach_dir: str = (
                        f"lab_attachments/{username}"
                    )
                    os.makedirs(lab_attach_dir, exist_ok=True)
                    lab_timestamp: str = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    lab_safe_name: str = (
                        f"{lab_timestamp}_{lab_uploaded_file.name}"
                    )
                    lab_file_path: str = os.path.join(
                        lab_attach_dir, lab_safe_name
                    )
                    with open(lab_file_path, "wb") as f:
                        f.write(lab_uploaded_file.getbuffer())
                    lab_test["attachment"] = {
                        "filename": lab_uploaded_file.name,
                        "filepath": lab_file_path,
                    }
                    logger.info(
                        "Lab attachment saved: %s", lab_file_path,
                    )

                st.session_state.lab_tests.append(lab_test)
                st.session_state.pending_lab_items = []
                save_user_data(username)
                st.session_state.lab_form_key += 1
                st.session_state.lab_item_key += 1
                st.session_state.lab_file_key += 1
                logger.info("Lab test saved: %s", lab_test)
                st.rerun()
    with col_cancel:
        if st.button(
            "Cancel", key="lab_cancel", use_container_width=True,
        ):
            st.session_state.pending_lab_items = []
            st.session_state.lab_form_key += 1
            st.session_state.lab_item_key += 1
            st.session_state.lab_file_key += 1
            st.rerun()

    # ── Records ──────────────────────────────────────────────────────────
    if st.session_state.lab_tests:
        st.markdown("---")
        st.markdown("#### Records")

        for idx, lt in enumerate(
            reversed(st.session_state.lab_tests)
        ):
            real_idx: int = (
                len(st.session_state.lab_tests) - 1 - idx
            )
            is_editing_lt: bool = (
                st.session_state.editing_lab_test == real_idx
            )

            with st.container(border=True):
                if is_editing_lt:
                    st.markdown("**Editing Lab Test**")

                    edited_lab_date = st.date_input(
                        "Date",
                        value=datetime.strptime(
                            lt["date"], "%Y-%m-%d"
                        ).date(),
                        key=f"edit_lab_date_{real_idx}",
                    )

                    # Editable test list stored in session state
                    edit_tests_key = f"edit_lab_tests_{real_idx}"
                    if edit_tests_key not in st.session_state:
                        st.session_state[edit_tests_key] = list(
                            lt.get("tests", [])
                        )

                    st.markdown("**Type**")
                    for t_idx, test_item in enumerate(
                        st.session_state[edit_tests_key]
                    ):
                        is_editing_ei: bool = (
                            st.session_state.editing_lab_item
                            == f"elt_{real_idx}_{t_idx}"
                        )
                        with st.container(border=True):
                            if is_editing_ei:
                                ei_cols = st.columns([2, 2, 2, 1, 1])
                                with ei_cols[0]:
                                    eet_test = st.text_input(
                                        "Test",
                                        value=test_item["test"],
                                        key=f"eet_test_{real_idx}_{t_idx}",
                                        label_visibility="collapsed",
                                    )
                                with ei_cols[1]:
                                    eet_result = st.text_input(
                                        "Result",
                                        value=test_item["result"],
                                        key=f"eet_result_{real_idx}_{t_idx}",
                                        label_visibility="collapsed",
                                    )
                                with ei_cols[2]:
                                    eet_normal = st.text_input(
                                        "Normal",
                                        value=test_item["normal"],
                                        key=f"eet_normal_{real_idx}_{t_idx}",
                                        label_visibility="collapsed",
                                    )
                                with ei_cols[3]:
                                    if st.button(
                                        "Save",
                                        key=f"eet_save_{real_idx}_{t_idx}",
                                        use_container_width=True,
                                    ):
                                        st.session_state[
                                            edit_tests_key
                                        ][t_idx] = {
                                            "test": eet_test.strip(),
                                            "result": eet_result.strip(),
                                            "normal": eet_normal.strip(),
                                        }
                                        st.session_state.editing_lab_item = (
                                            None
                                        )
                                        st.rerun()
                                with ei_cols[4]:
                                    if st.button(
                                        "Cancel",
                                        key=f"eet_cancel_{real_idx}_{t_idx}",
                                        use_container_width=True,
                                    ):
                                        st.session_state.editing_lab_item = (
                                            None
                                        )
                                        st.rerun()
                            else:
                                t_col_test, t_col_sep1, t_col_result, t_col_sep2, t_col_normal, t_col_edit, t_col_trash = (
                                    st.columns([2, 0.15, 2, 0.15, 2, 0.5, 0.5])
                                )
                                with t_col_test:
                                    st.markdown(
                                        f"**{test_item['test']}**"
                                    )
                                with t_col_sep1:
                                    st.markdown(
                                        "<div style='border-left: 2px solid"
                                        " #ccc; height: 100%;"
                                        " min-height: 30px;'></div>",
                                        unsafe_allow_html=True,
                                    )
                                with t_col_result:
                                    st.markdown(test_item["result"])
                                with t_col_sep2:
                                    st.markdown(
                                        "<div style='border-left: 2px solid"
                                        " #ccc; height: 100%;"
                                        " min-height: 30px;'></div>",
                                        unsafe_allow_html=True,
                                    )
                                with t_col_normal:
                                    st.markdown(test_item["normal"])
                                with t_col_edit:
                                    if st.button(
                                        "\u270F\uFE0F",
                                        key=f"epen_lab_{real_idx}_{t_idx}",
                                    ):
                                        st.session_state.editing_lab_item = (
                                            f"elt_{real_idx}_{t_idx}"
                                        )
                                        st.rerun()
                                with t_col_trash:
                                    if st.button(
                                        "\U0001f5d1\uFE0F",
                                        key=f"etrash_lab_{real_idx}_{t_idx}",
                                    ):
                                        st.session_state[
                                            edit_tests_key
                                        ].pop(t_idx)
                                        st.rerun()

                    edit_add_cols = st.columns([2, 2, 2, 1])
                    with edit_add_cols[0]:
                        edit_new_test = st.text_input(
                            "Test",
                            key=f"edit_lab_newtest_{real_idx}",
                            label_visibility="collapsed",
                            placeholder="Test",
                        )
                    with edit_add_cols[1]:
                        edit_new_result = st.text_input(
                            "Result",
                            key=f"edit_lab_newresult_{real_idx}",
                            label_visibility="collapsed",
                            placeholder="Result",
                        )
                    with edit_add_cols[2]:
                        edit_new_normal = st.text_input(
                            "Normal",
                            key=f"edit_lab_newnormal_{real_idx}",
                            label_visibility="collapsed",
                            placeholder="Normal",
                        )
                    with edit_add_cols[3]:
                        if st.button(
                            "Add",
                            key=f"edit_lab_add_{real_idx}",
                            use_container_width=True,
                        ):
                            if (
                                edit_new_test.strip()
                                and edit_new_result.strip()
                                and edit_new_normal.strip()
                            ):
                                st.session_state[edit_tests_key].append({
                                    "test": edit_new_test.strip(),
                                    "result": edit_new_result.strip(),
                                    "normal": edit_new_normal.strip(),
                                })
                                st.rerun()

                    edited_lab_doctor = st.text_input(
                        "Doctor",
                        value=lt["doctor"],
                        key=f"edit_lab_doctor_{real_idx}",
                    )

                    st.caption(
                        "💡 Formatting: `**bold**`, `*italic*`, "
                        "`- bullet point`, `1. numbered list`"
                    )
                    edited_lab_notes = st.text_area(
                        "Notes",
                        value=lt.get("notes", ""),
                        height=120,
                        key=f"edit_lab_notes_{real_idx}",
                    )

                    if lt.get("attachment"):
                        st.caption(
                            f"Current attachment: "
                            f"{lt['attachment']['filename']}"
                        )
                    edit_lab_file = st.file_uploader(
                        "Replace PDF (optional)",
                        type=["pdf"],
                        key=f"edit_lab_file_{real_idx}",
                    )

                    col_save_edit, col_cancel_edit = st.columns(2)
                    with col_save_edit:
                        if st.button(
                            "Save Changes",
                            key=f"save_lab_{real_idx}",
                            use_container_width=True,
                        ):
                            updated_lab: dict[str, Any] = {
                                "date": edited_lab_date.strftime("%Y-%m-%d"),
                                "tests": list(
                                    st.session_state[edit_tests_key]
                                ),
                                "doctor": edited_lab_doctor.strip(),
                                "notes": edited_lab_notes.strip(),
                            }

                            if edit_lab_file is not None:
                                lab_attach_dir = (
                                    f"lab_attachments/{username}"
                                )
                                os.makedirs(lab_attach_dir, exist_ok=True)
                                lab_ts: str = datetime.now().strftime(
                                    "%Y%m%d_%H%M%S"
                                )
                                lab_sname: str = (
                                    f"{lab_ts}_{edit_lab_file.name}"
                                )
                                lab_fpath: str = os.path.join(
                                    lab_attach_dir, lab_sname
                                )
                                with open(lab_fpath, "wb") as f:
                                    f.write(edit_lab_file.getbuffer())
                                updated_lab["attachment"] = {
                                    "filename": edit_lab_file.name,
                                    "filepath": lab_fpath,
                                }
                                logger.info(
                                    "Lab attachment updated: %s",
                                    lab_fpath,
                                )
                            elif lt.get("attachment"):
                                updated_lab["attachment"] = lt["attachment"]

                            st.session_state.lab_tests[real_idx] = (
                                updated_lab
                            )
                            del st.session_state[edit_tests_key]
                            save_user_data(username)
                            st.session_state.editing_lab_test = None
                            logger.info("Lab test updated")
                            st.rerun()
                    with col_cancel_edit:
                        if st.button(
                            "Cancel",
                            key=f"cancel_lab_{real_idx}",
                            use_container_width=True,
                        ):
                            if edit_tests_key in st.session_state:
                                del st.session_state[edit_tests_key]
                            st.session_state.editing_lab_test = None
                            st.rerun()
                else:
                    col_date, col_doctor, col_spacer = st.columns(
                        [2, 3, 1]
                    )

                    with col_date:
                        parsed_date = datetime.strptime(
                            lt["date"], "%Y-%m-%d"
                        )
                        display_date = (
                            f"{parsed_date.strftime('%B')} "
                            f"{parsed_date.day}, "
                            f"{parsed_date.year}"
                        )
                        st.markdown(f"**{display_date}**")
                    with col_doctor:
                        st.markdown(f"**Dr.** {lt['doctor']}")

                    # Test list
                    for test_item in lt.get("tests", []):
                        with st.container(border=True):
                            t_col_test, t_col_sep1, t_col_result, t_col_sep2, t_col_normal = (
                                st.columns([2, 0.15, 2, 0.15, 2])
                            )
                            with t_col_test:
                                st.markdown(
                                    f"**{test_item['test']}**"
                                )
                            with t_col_sep1:
                                st.markdown(
                                    "<div style='border-left: 2px solid"
                                    " #ccc; height: 100%;"
                                    " min-height: 30px;'></div>",
                                    unsafe_allow_html=True,
                                )
                            with t_col_result:
                                st.markdown(test_item["result"])
                            with t_col_sep2:
                                st.markdown(
                                    "<div style='border-left: 2px solid"
                                    " #ccc; height: 100%;"
                                    " min-height: 30px;'></div>",
                                    unsafe_allow_html=True,
                                )
                            with t_col_normal:
                                st.markdown(test_item["normal"])

                    if lt.get("notes"):
                        st.markdown("---")
                        st.markdown(lt["notes"])

                    if lt.get("attachment"):
                        st.markdown("---")
                        st.caption(
                            f"📄 {lt['attachment']['filename']}"
                        )
                        with open(
                            lt["attachment"]["filepath"], "rb"
                        ) as f:
                            st.download_button(
                                label="Download PDF",
                                data=f,
                                file_name=lt["attachment"]["filename"],
                                mime="application/pdf",
                                key=f"dl_lab_{real_idx}",
                            )

                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(
                            "Edit",
                            key=f"edit_lab_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_lab_test = real_idx
                            st.rerun()
                    with col_delete:
                        if st.button(
                            "Delete",
                            key=f"del_lab_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.lab_tests.pop(real_idx)
                            save_user_data(username)
                            logger.info("Lab test deleted")
                            st.rerun()
    else:
        st.info(
            "No lab tests yet. Start tracking your lab results!"
        )
