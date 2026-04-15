"""Blood Pressure tab UI — tracking and trends."""

import logging
from datetime import datetime
from typing import Any

import streamlit as st

from core.user_data import save_user_data

logger = logging.getLogger(__name__)


def render(username: str) -> None:
    """Render the Blood Pressure tab.

    Args:
        username: The authenticated username.
    """
    st.markdown("### Blood Pressure Tracker")
    st.caption("Record and monitor your blood pressure over time")

    # ── Entry Form ───────────────────────────────────────────────────────
    with st.form("bp_form", clear_on_submit=True):
        bp_cols = st.columns([1, 1, 1, 1, 1])

        with bp_cols[0]:
            bp_date = st.date_input("Date", key="bp_date")
        with bp_cols[1]:
            bp_time = st.time_input("Time", key="bp_time")
        with bp_cols[2]:
            bp_systolic = st.number_input(
                "Systolic", min_value=60, max_value=250, value=120,
                key="bp_systolic",
            )
        with bp_cols[3]:
            bp_diastolic = st.number_input(
                "Diastolic", min_value=30, max_value=150, value=80,
                key="bp_diastolic",
            )
        with bp_cols[4]:
            bp_pulse = st.number_input(
                "Pulse", min_value=30, max_value=220, value=72,
                key="bp_pulse",
            )

        if st.form_submit_button(
            "Save Reading", use_container_width=True,
        ):
            reading: dict[str, Any] = {
                "date": bp_date.strftime("%Y-%m-%d"),
                "time": bp_time.strftime("%H:%M"),
                "systolic": bp_systolic,
                "diastolic": bp_diastolic,
                "pulse": bp_pulse,
            }
            st.session_state.bp_readings.append(reading)
            save_user_data(username)
            logger.info("Blood pressure reading saved: %s", reading)
            st.rerun()

    # ── Chart ────────────────────────────────────────────────────────────
    if st.session_state.bp_readings:
        st.markdown("---")
        st.markdown("#### Trends")

        import pandas as pd
        import plotly.graph_objects as go

        df = pd.DataFrame(st.session_state.bp_readings)
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
        df = df.sort_values("datetime")
        df["label"] = df["datetime"].dt.strftime("%b %d, %Y")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Systolic",
            x=df["label"],
            y=df["systolic"],
            marker_color="#EF4444",
        ))
        fig.add_trace(go.Bar(
            name="Diastolic",
            x=df["label"],
            y=df["diastolic"],
            marker_color="#3B82F6",
        ))
        fig.add_trace(go.Bar(
            name="Pulse",
            x=df["label"],
            y=df.get("pulse", pd.Series([0] * len(df))),
            marker_color="#10B981",
        ))
        fig.update_layout(
            barmode="group",
            xaxis_title="Date",
            yaxis_title="mmHg / BPM",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=40, b=40),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Readings ──────────────────────────────────────────────────────
        st.markdown("#### Readings")

        for idx, reading in enumerate(
            reversed(st.session_state.bp_readings)
        ):
            real_idx: int = (
                len(st.session_state.bp_readings) - 1 - idx
            )

            with st.container(border=True):
                col_date, col_sys, col_dia, col_pulse, col_del = st.columns(
                    [2, 1.5, 1.5, 1.5, 1]
                )

                with col_date:
                    parsed_date = datetime.strptime(
                        reading["date"], "%Y-%m-%d"
                    )
                    display_date = (
                        f"{parsed_date.strftime('%B')} "
                        f"{parsed_date.day}, "
                        f"{parsed_date.year}"
                    )
                    st.markdown(
                        f"**{display_date}**  \n"
                        f"{reading.get('time', '')}"
                    )
                with col_sys:
                    st.metric(
                        label="Systolic",
                        value=f"{reading['systolic']}",
                    )
                with col_dia:
                    st.metric(
                        label="Diastolic",
                        value=f"{reading['diastolic']}",
                    )
                with col_pulse:
                    st.metric(
                        label="Pulse",
                        value=f"{reading.get('pulse', '—')}",
                    )
                with col_del:
                    st.markdown("")
                    if st.button(
                        "Delete",
                        key=f"del_bp_{real_idx}",
                        use_container_width=True,
                    ):
                        st.session_state.bp_readings.pop(real_idx)
                        save_user_data(username)
                        logger.info("Blood pressure reading deleted")
                        st.rerun()
    else:
        st.info(
            "No readings yet. Start tracking your blood pressure!"
        )
