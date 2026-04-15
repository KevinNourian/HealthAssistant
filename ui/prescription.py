"""Prescription tab UI — prescription tracking with drug/dose builder."""

import logging
from datetime import datetime
from typing import Any

import streamlit as st

from core.user_data import save_user_data

logger = logging.getLogger(__name__)


def render(username: str) -> None:
    """Render the Prescription tab.

    Args:
        username: The authenticated username.
    """
    st.markdown("### Prescription")
    st.caption("Record and track your prescriptions")

    # ── Entry Form ───────────────────────────────────────────────────────
    rx_key = st.session_state.rx_form_key
    drug_key = st.session_state.rx_drug_key

    rx_date = st.date_input("Date", key=f"rx_date_{rx_key}")

    # ── Drug / Dose builder ──────────────────────────────────────────────
    st.markdown("**Drugs & Doses**")

    # Display already-added drugs
    for d_idx, drug_entry in enumerate(
        st.session_state.pending_rx_drugs
    ):
        is_editing_drug: bool = (
            st.session_state.editing_rx_drug
            == f"pending_{d_idx}"
        )
        with st.container(border=True):
            if is_editing_drug:
                ed_cols = st.columns([2, 2, 1, 1])
                with ed_cols[0]:
                    edited_drug = st.text_input(
                        "Drug",
                        value=drug_entry["drug"],
                        key=f"pedit_drug_{d_idx}_{rx_key}",
                        label_visibility="collapsed",
                    )
                with ed_cols[1]:
                    edited_dose = st.text_input(
                        "Dose",
                        value=drug_entry["dose"],
                        key=f"pedit_dose_{d_idx}_{rx_key}",
                        label_visibility="collapsed",
                    )
                with ed_cols[2]:
                    if st.button(
                        "Save",
                        key=f"pedit_save_{d_idx}_{rx_key}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_rx_drugs[d_idx] = {
                            "drug": edited_drug.strip(),
                            "dose": edited_dose.strip(),
                        }
                        st.session_state.editing_rx_drug = None
                        st.rerun()
                with ed_cols[3]:
                    if st.button(
                        "Cancel",
                        key=f"pedit_cancel_{d_idx}_{rx_key}",
                        use_container_width=True,
                    ):
                        st.session_state.editing_rx_drug = None
                        st.rerun()
            else:
                d_col_drug, d_col_sep, d_col_dose, d_col_edit, d_col_trash = (
                    st.columns([2, 0.15, 2, 0.5, 0.5])
                )
                with d_col_drug:
                    st.markdown(f"**{drug_entry['drug']}**")
                with d_col_sep:
                    st.markdown(
                        "<div style='border-left: 2px solid #ccc;"
                        " height: 100%; min-height: 30px;'></div>",
                        unsafe_allow_html=True,
                    )
                with d_col_dose:
                    st.markdown(drug_entry["dose"])
                with d_col_edit:
                    if st.button(
                        "\u270F\uFE0F",
                        key=f"pen_rx_{d_idx}_{rx_key}",
                    ):
                        st.session_state.editing_rx_drug = (
                            f"pending_{d_idx}"
                        )
                        st.rerun()
                with d_col_trash:
                    if st.button(
                        "\U0001f5d1\uFE0F",
                        key=f"trash_rx_{d_idx}_{rx_key}",
                    ):
                        st.session_state.pending_rx_drugs.pop(d_idx)
                        st.rerun()

    # Input row for new drug/dose
    add_cols = st.columns([2, 2, 1])
    with add_cols[0]:
        new_drug = st.text_input(
            "Drug", key=f"rx_drug_{drug_key}", label_visibility="collapsed",
            placeholder="Drug name",
        )
    with add_cols[1]:
        new_dose = st.text_input(
            "Dose", key=f"rx_dose_{drug_key}", label_visibility="collapsed",
            placeholder="Dose",
        )
    with add_cols[2]:
        if st.button("Add", key=f"rx_add_{drug_key}",
                      use_container_width=True):
            if new_drug.strip() and new_dose.strip():
                st.session_state.pending_rx_drugs.append({
                    "drug": new_drug.strip(),
                    "dose": new_dose.strip(),
                })
                st.session_state.rx_drug_key += 1
                st.rerun()
            else:
                st.warning("Please enter both drug and dose.")

    rx_doctor = st.text_input("Doctor", key=f"rx_doctor_{rx_key}")

    st.caption(
        "💡 Formatting: `**bold**`, `*italic*`, "
        "`- bullet point`, `1. numbered list`"
    )
    rx_notes = st.text_area(
        "Notes",
        key=f"rx_notes_{rx_key}",
        height=120,
    )

    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Save Prescription", use_container_width=True):
            if not st.session_state.pending_rx_drugs:
                st.warning("Please add at least one drug and dose.")
            elif not rx_doctor.strip():
                st.warning("Please enter a doctor name.")
            else:
                prescription: dict[str, Any] = {
                    "date": rx_date.strftime("%Y-%m-%d"),
                    "drugs": list(st.session_state.pending_rx_drugs),
                    "doctor": rx_doctor.strip(),
                    "notes": rx_notes.strip(),
                }
                st.session_state.prescriptions.append(prescription)
                st.session_state.pending_rx_drugs = []
                save_user_data(username)
                st.session_state.rx_form_key += 1
                st.session_state.rx_drug_key += 1
                logger.info("Prescription saved: %s", prescription)
                st.rerun()
    with col_cancel:
        if st.button(
            "Cancel", key="rx_cancel", use_container_width=True,
        ):
            st.session_state.pending_rx_drugs = []
            st.session_state.rx_form_key += 1
            st.session_state.rx_drug_key += 1
            st.rerun()

    # ── Records ──────────────────────────────────────────────────────────
    if st.session_state.prescriptions:
        st.markdown("---")
        st.markdown("#### Records")

        for idx, rx in enumerate(
            reversed(st.session_state.prescriptions)
        ):
            real_idx: int = (
                len(st.session_state.prescriptions) - 1 - idx
            )
            is_editing_rx: bool = (
                st.session_state.editing_prescription == real_idx
            )

            with st.container(border=True):
                if is_editing_rx:
                    st.markdown("**Editing Prescription**")

                    edited_rx_date = st.date_input(
                        "Date",
                        value=datetime.strptime(
                            rx["date"], "%Y-%m-%d"
                        ).date(),
                        key=f"edit_rx_date_{real_idx}",
                    )

                    # Editable drug list stored in session state
                    edit_drugs_key = f"edit_rx_drugs_{real_idx}"
                    if edit_drugs_key not in st.session_state:
                        st.session_state[edit_drugs_key] = list(
                            rx.get("drugs", [])
                        )

                    st.markdown("**Drugs & Doses**")
                    for d_idx, drug_entry in enumerate(
                        st.session_state[edit_drugs_key]
                    ):
                        is_editing_ed: bool = (
                            st.session_state.editing_rx_drug
                            == f"erx_{real_idx}_{d_idx}"
                        )
                        with st.container(border=True):
                            if is_editing_ed:
                                ed_cols = st.columns([2, 2, 1, 1])
                                with ed_cols[0]:
                                    eed_drug = st.text_input(
                                        "Drug",
                                        value=drug_entry["drug"],
                                        key=f"eed_drug_{real_idx}_{d_idx}",
                                        label_visibility="collapsed",
                                    )
                                with ed_cols[1]:
                                    eed_dose = st.text_input(
                                        "Dose",
                                        value=drug_entry["dose"],
                                        key=f"eed_dose_{real_idx}_{d_idx}",
                                        label_visibility="collapsed",
                                    )
                                with ed_cols[2]:
                                    if st.button(
                                        "Save",
                                        key=f"eed_save_{real_idx}_{d_idx}",
                                        use_container_width=True,
                                    ):
                                        st.session_state[
                                            edit_drugs_key
                                        ][d_idx] = {
                                            "drug": eed_drug.strip(),
                                            "dose": eed_dose.strip(),
                                        }
                                        st.session_state.editing_rx_drug = (
                                            None
                                        )
                                        st.rerun()
                                with ed_cols[3]:
                                    if st.button(
                                        "Cancel",
                                        key=f"eed_cancel_{real_idx}_{d_idx}",
                                        use_container_width=True,
                                    ):
                                        st.session_state.editing_rx_drug = (
                                            None
                                        )
                                        st.rerun()
                            else:
                                d_col_drug, d_col_sep, d_col_dose, d_col_edit, d_col_trash = (
                                    st.columns([2, 0.15, 2, 0.5, 0.5])
                                )
                                with d_col_drug:
                                    st.markdown(
                                        f"**{drug_entry['drug']}**"
                                    )
                                with d_col_sep:
                                    st.markdown(
                                        "<div style='border-left: 2px solid"
                                        " #ccc; height: 100%;"
                                        " min-height: 30px;'></div>",
                                        unsafe_allow_html=True,
                                    )
                                with d_col_dose:
                                    st.markdown(drug_entry["dose"])
                                with d_col_edit:
                                    if st.button(
                                        "\u270F\uFE0F",
                                        key=f"epen_rx_{real_idx}_{d_idx}",
                                    ):
                                        st.session_state.editing_rx_drug = (
                                            f"erx_{real_idx}_{d_idx}"
                                        )
                                        st.rerun()
                                with d_col_trash:
                                    if st.button(
                                        "\U0001f5d1\uFE0F",
                                        key=f"etrash_rx_{real_idx}_{d_idx}",
                                    ):
                                        st.session_state[
                                            edit_drugs_key
                                        ].pop(d_idx)
                                        st.rerun()

                    edit_add_cols = st.columns([2, 2, 1])
                    with edit_add_cols[0]:
                        edit_new_drug = st.text_input(
                            "Drug",
                            key=f"edit_rx_newdrug_{real_idx}",
                            label_visibility="collapsed",
                            placeholder="Drug name",
                        )
                    with edit_add_cols[1]:
                        edit_new_dose = st.text_input(
                            "Dose",
                            key=f"edit_rx_newdose_{real_idx}",
                            label_visibility="collapsed",
                            placeholder="Dose",
                        )
                    with edit_add_cols[2]:
                        if st.button(
                            "Add",
                            key=f"edit_rx_add_{real_idx}",
                            use_container_width=True,
                        ):
                            if (
                                edit_new_drug.strip()
                                and edit_new_dose.strip()
                            ):
                                st.session_state[edit_drugs_key].append({
                                    "drug": edit_new_drug.strip(),
                                    "dose": edit_new_dose.strip(),
                                })
                                st.rerun()

                    edited_rx_doctor = st.text_input(
                        "Doctor",
                        value=rx["doctor"],
                        key=f"edit_rx_doctor_{real_idx}",
                    )

                    st.caption(
                        "💡 Formatting: `**bold**`, `*italic*`, "
                        "`- bullet point`, `1. numbered list`"
                    )
                    edited_rx_notes = st.text_area(
                        "Notes",
                        value=rx.get("notes", ""),
                        height=120,
                        key=f"edit_rx_notes_{real_idx}",
                    )

                    col_save_edit, col_cancel_edit = st.columns(2)
                    with col_save_edit:
                        if st.button(
                            "Save Changes",
                            key=f"save_rx_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.prescriptions[real_idx] = {
                                "date": edited_rx_date.strftime("%Y-%m-%d"),
                                "drugs": list(
                                    st.session_state[edit_drugs_key]
                                ),
                                "doctor": edited_rx_doctor.strip(),
                                "notes": edited_rx_notes.strip(),
                            }
                            del st.session_state[edit_drugs_key]
                            save_user_data(username)
                            st.session_state.editing_prescription = None
                            logger.info("Prescription updated")
                            st.rerun()
                    with col_cancel_edit:
                        if st.button(
                            "Cancel",
                            key=f"cancel_rx_{real_idx}",
                            use_container_width=True,
                        ):
                            if edit_drugs_key in st.session_state:
                                del st.session_state[edit_drugs_key]
                            st.session_state.editing_prescription = None
                            st.rerun()
                else:
                    col_date, col_doctor, col_spacer = st.columns(
                        [2, 3, 1]
                    )

                    with col_date:
                        parsed_date = datetime.strptime(
                            rx["date"], "%Y-%m-%d"
                        )
                        display_date = (
                            f"{parsed_date.strftime('%B')} "
                            f"{parsed_date.day}, "
                            f"{parsed_date.year}"
                        )
                        st.markdown(f"**{display_date}**")
                    with col_doctor:
                        st.markdown(f"**Dr.** {rx['doctor']}")

                    # Drug list
                    for drug_entry in rx.get("drugs", []):
                        with st.container(border=True):
                            d_col_drug, d_col_sep, d_col_dose = st.columns(
                                [2, 0.15, 2]
                            )
                            with d_col_drug:
                                st.markdown(
                                    f"**{drug_entry['drug']}**"
                                )
                            with d_col_sep:
                                st.markdown(
                                    "<div style='border-left: 2px solid"
                                    " #ccc; height: 100%;"
                                    " min-height: 30px;'></div>",
                                    unsafe_allow_html=True,
                                )
                            with d_col_dose:
                                st.markdown(drug_entry["dose"])

                    if rx.get("notes"):
                        st.markdown("---")
                        st.markdown(rx["notes"])

                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(
                            "Edit",
                            key=f"edit_rx_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_prescription = (
                                real_idx
                            )
                            st.rerun()
                    with col_delete:
                        if st.button(
                            "Delete",
                            key=f"del_rx_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.prescriptions.pop(real_idx)
                            save_user_data(username)
                            logger.info("Prescription deleted")
                            st.rerun()
    else:
        st.info(
            "No prescriptions yet. Start tracking your prescriptions!"
        )
