"""
Authentication setup and login UI for the Health Assistant.

Uses ``streamlit-authenticator`` backed by a ``credentials.yaml`` file
to provide cookie-based session management.  Includes a self-service
registration form.
"""

import logging
import os
import re

import bcrypt
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

logger = logging.getLogger(__name__)

_CREDS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "credentials.yaml",
)


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
            "Missing `credentials.yaml`. Please create the file with valid credentials."
        )
        st.stop()

    return stauth.Authenticate(
        config_auth["credentials"],
        config_auth["cookie"]["name"],
        config_auth["cookie"]["key"],
        config_auth["cookie"]["expiry_days"],
    )


def _load_credentials() -> dict:
    """Load the raw credentials dictionary from YAML.

    Returns:
        The full YAML configuration dictionary.
    """
    with open(_CREDS_PATH) as file:
        return yaml.load(file, Loader=SafeLoader)


def _save_credentials(config_auth: dict) -> None:
    """Write the credentials dictionary back to YAML.

    Args:
        config_auth: The full YAML configuration dictionary.
    """
    with open(_CREDS_PATH, "w") as file:
        yaml.dump(config_auth, file, default_flow_style=False)
    logger.info("Credentials file updated")


def _hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt.

    Args:
        password: The plaintext password.

    Returns:
        The bcrypt-hashed password string.
    """
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")


def _validate_registration(
    name: str,
    username: str,
    email: str,
    password: str,
    password_confirm: str,
    existing_usernames: set[str],
    existing_emails: set[str],
) -> str | None:
    """Validate registration form inputs.

    Args:
        name: Display name.
        username: Desired username.
        email: Email address.
        password: Password.
        password_confirm: Password confirmation.
        existing_usernames: Set of taken usernames.
        existing_emails: Set of taken emails.

    Returns:
        An error message string, or ``None`` if validation passes.
    """
    if not all([name, username, email, password, password_confirm]):
        return "All fields are required."
    if len(username) < 3:
        return "Username must be at least 3 characters."
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return "Username can only contain letters, numbers, and underscores."
    if username.lower() in existing_usernames:
        return "Username is already taken."
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        return "Please enter a valid email address."
    if email.lower() in existing_emails:
        return "Email is already registered."
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if password != password_confirm:
        return "Passwords do not match."
    return None


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
    # Initialize auth page state
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "login"

    with st.sidebar:
        st.markdown("## 🏥 Health Assistant")
        st.markdown("---")

        if st.session_state.auth_page == "login":
            st.markdown("##### Sign In")

            authenticator.login(location="sidebar")

            if st.session_state.get("authentication_status") is False:
                logger.warning("Failed login attempt")
                st.error("Invalid username or password")

            st.markdown("")
            if st.button(
                "Create an Account",
                use_container_width=True,
                key="go_to_register",
            ):
                st.session_state.auth_page = "register"
                st.rerun()

        else:
            _render_registration_form()

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

    logger.info("User authenticated: %s", st.session_state.get("username"))
    st.rerun()


def _render_registration_form() -> None:
    """Render the registration form in the sidebar."""
    st.markdown("##### Create Account")

    with st.form("register_form", clear_on_submit=True):
        reg_name = st.text_input("Name", placeholder="Your full name")
        reg_username = st.text_input(
            "Username", placeholder="Choose a username",
        )
        reg_email = st.text_input(
            "Email", placeholder="you@example.com",
        )
        reg_password = st.text_input(
            "Password", type="password", placeholder="Min 8 characters",
        )
        reg_password_confirm = st.text_input(
            "Confirm Password", type="password",
        )

        submitted = st.form_submit_button(
            "Register", use_container_width=True,
        )

    if submitted:
        config_auth = _load_credentials()
        existing_usernames = {
            u.lower()
            for u in config_auth["credentials"]["usernames"]
        }
        existing_emails = {
            v.get("email", "").lower()
            for v in config_auth["credentials"]["usernames"].values()
        }

        error = _validate_registration(
            reg_name,
            reg_username,
            reg_email,
            reg_password,
            reg_password_confirm,
            existing_usernames,
            existing_emails,
        )

        if error:
            st.error(error)
        else:
            config_auth["credentials"]["usernames"][reg_username.lower()] = {
                "email": reg_email,
                "name": reg_name,
                "password": _hash_password(reg_password),
            }
            _save_credentials(config_auth)
            logger.info("New user registered: %s", reg_username)
            st.success("Account created! Please sign in.")
            st.session_state.auth_page = "login"
            st.rerun()

    st.markdown("")
    if st.button(
        "Back to Sign In",
        use_container_width=True,
        key="go_to_login",
    ):
        st.session_state.auth_page = "login"
        st.rerun()
