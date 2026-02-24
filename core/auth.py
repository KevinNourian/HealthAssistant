"""
Authentication setup and login UI for the Health Assistant.

Uses ``streamlit-authenticator`` backed by a ``credentials.yaml`` file
to provide cookie-based session management.
"""

import logging
import os

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

logger = logging.getLogger(__name__)

_CREDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "credentials.yaml")


def load_authenticator() -> stauth.Authenticate:
    """Load credentials from YAML and create the authenticator instance.

    Returns:
        A configured ``stauth.Authenticate`` object ready for use.

    Raises:
        FileNotFoundError: Logged and surfaced in the Streamlit UI;
            also calls ``st.stop()`` so the app halts gracefully.
    """
    try:
        with open(_CREDS_PATH) as file:
            config_auth: dict = yaml.load(file, Loader=SafeLoader)
        logger.info("Credentials loaded successfully")
    except FileNotFoundError:
        logger.error("credentials.yaml not found")
        st.error(
            "Missing `credentials.yaml`. "
            "Please create the file with valid credentials."
        )
        st.stop()

    return stauth.Authenticate(
        config_auth["credentials"],
        config_auth["cookie"]["name"],
        config_auth["cookie"]["key"],
        config_auth["cookie"]["expiry_days"],
    )


def render_login(authenticator: stauth.Authenticate) -> None:
    """Render the login sidebar and landing page for unauthenticated users.

    This function calls ``st.stop()`` if the user has not yet
    authenticated, preventing the rest of the app from rendering.
    On successful login it triggers ``st.rerun()`` so the main app
    loads immediately.

    Args:
        authenticator: The ``stauth.Authenticate`` instance that manages
            the login widget and session cookie.
    """
    with st.sidebar:
        st.markdown("## 🏥 Health Assistant")
        st.markdown("---")
        st.markdown("##### Sign In")

        authenticator.login(location="sidebar")

        if st.session_state.get("authentication_status") is False:
            logger.warning("Failed login attempt")
            st.error("Invalid username or password")
        elif st.session_state.get("authentication_status") is None:
            st.markdown("")
            st.markdown(
                """
                <div style="background: #F5F5F5; padding: 12px;
                            border-radius: 8px; font-size: 13px;">
                    <strong>Demo Accounts</strong><br>
                    <span style="color: #6B7280;">alice / temp123</span><br>
                    <span style="color: #6B7280;">bob / temp456</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.session_state.get("authentication_status") is not True:
        st.markdown(
            """
            <div style="text-align: center; padding: 4rem 2rem;">
                <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">
                    Health Assistant
                </h1>
                <p style="color: #6B7280; font-size: 1.125rem;">
                    Your personal health companion
                </p>
                <p style="color: #9CA3AF; font-size: 0.875rem;
                   margin-top: 2rem;">
                    ← Please sign in to continue
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    logger.info(
        "User authenticated: %s", st.session_state.get("username")
    )
    st.rerun()
