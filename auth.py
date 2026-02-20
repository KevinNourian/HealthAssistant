"""
Authentication setup and login UI for the Health Assistant.
"""

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


def load_authenticator() -> stauth.Authenticate:
    """Load credentials and create the authenticator instance."""
    try:
        with open('credentials.yaml') as file:
            config_auth = yaml.load(file, Loader=SafeLoader)
    except FileNotFoundError:
        st.error(
            "Missing `credentials.yaml`. "
            "Please create the file with valid credentials."
        )
        st.stop()

    return stauth.Authenticate(
        config_auth['credentials'],
        config_auth['cookie']['name'],
        config_auth['cookie']['key'],
        config_auth['cookie']['expiry_days']
    )


def render_login(authenticator: stauth.Authenticate) -> None:
    """Render the login sidebar and landing page for unauthenticated users."""
    with st.sidebar:
        st.markdown("## 🏥 Health Assistant")
        st.markdown("---")
        st.markdown("##### Sign In")

        authenticator.login(location='sidebar')

        if st.session_state.get("authentication_status") is False:
            st.error('Invalid username or password')
        elif st.session_state.get("authentication_status") is None:
            st.markdown("")
            st.markdown("""
            <div style="background: #F5F5F5; padding: 12px; border-radius: 8px;
                        font-size: 13px;">
                <strong>Demo Accounts</strong><br>
                <span style="color: #6B7280;">alice / temp123</span><br>
                <span style="color: #6B7280;">bob / temp456</span>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.get("authentication_status") is not True:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">
                Health Assistant
            </h1>
            <p style="color: #6B7280; font-size: 1.125rem;">
                Your personal health companion
            </p>
            <p style="color: #9CA3AF; font-size: 0.875rem; margin-top: 2rem;">
                ← Please sign in to continue
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.rerun()
