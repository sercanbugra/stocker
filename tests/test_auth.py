"""
Comprehensive tests for the Stocker auth system:
  - Email registration  (/register/email)
  - Email login         (/login/email)
  - Logout              (/logout)
  - Session persistence
  - Watchlist auth gate (/api/watchlist)
  - Google OAuth flow   (mocked)
"""
import json
import os
import sys
import tempfile
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Prevent real OAuth creds from accidentally activating during tests
os.environ.setdefault("GOOGLE_OAUTH_CLIENT_ID", "")
os.environ.setdefault("GOOGLE_OAUTH_CLIENT_SECRET", "")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key")


@pytest.fixture()
def tmp_data_dir(tmp_path, monkeypatch):
    """Route all file I/O to a throwaway directory."""
    monkeypatch.setenv("PERSISTENT_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture()
def app(tmp_data_dir):
    """Create a fresh Flask test application with isolated data dir."""
    # Re-import so _DATA_DIR picks up the monkeypatched env var
    import importlib
    import app as app_module
    importlib.reload(app_module)

    app_module.app.config["TESTING"] = True
    app_module.app.config["SECRET_KEY"] = "test-secret-key"
    # Disable CSRF / session signing complexity in tests
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    yield app_module.app


@pytest.fixture()
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def post_json(client, url, payload):
    return client.post(
        url,
        data=json.dumps(payload),
        content_type="application/json",
    )


def register(client, email="user@example.com", password="secret123"):
    return post_json(client, "/register/email", {"email": email, "password": password})


def login(client, email="user@example.com", password="secret123"):
    return post_json(client, "/login/email", {"email": email, "password": password})


# ===========================================================================
# REGISTRATION
# ===========================================================================

class TestRegister:
    def test_register_success(self, client):
        rv = register(client)
        assert rv.status_code == 200
        body = rv.get_json()
        assert body.get("ok") is True

    def test_register_sets_session(self, client, app):
        register(client)
        with client.session_transaction() as sess:
            assert sess.get("user_email") == "user@example.com"

    def test_register_duplicate_email(self, client):
        register(client)
        rv = register(client)  # second call — same email
        assert rv.status_code == 409
        assert "already exists" in rv.get_json().get("error", "").lower()

    def test_register_email_case_insensitive(self, client):
        """Emails should be normalised to lowercase; UPPER@example.com == upper@example.com."""
        rv1 = post_json(client, "/register/email", {"email": "UPPER@example.com", "password": "secret123"})
        assert rv1.status_code == 200
        rv2 = post_json(client, "/register/email", {"email": "upper@example.com", "password": "secret123"})
        assert rv2.status_code == 409

    def test_register_invalid_email(self, client):
        rv = post_json(client, "/register/email", {"email": "not-an-email", "password": "secret123"})
        assert rv.status_code == 400
        assert "invalid email" in rv.get_json().get("error", "").lower()

    def test_register_missing_email(self, client):
        rv = post_json(client, "/register/email", {"password": "secret123"})
        assert rv.status_code == 400

    def test_register_missing_password(self, client):
        rv = post_json(client, "/register/email", {"email": "user@example.com"})
        assert rv.status_code == 400

    def test_register_empty_body(self, client):
        rv = post_json(client, "/register/email", {})
        assert rv.status_code == 400

    def test_register_non_json_body(self, client):
        rv = client.post("/register/email", data="not-json", content_type="text/plain")
        assert rv.status_code == 400

    def test_register_password_too_short(self, client):
        rv = post_json(client, "/register/email", {"email": "user@example.com", "password": "abc"})
        assert rv.status_code == 400
        assert "6" in rv.get_json().get("error", "")

    def test_register_password_exactly_6_chars(self, client):
        rv = post_json(client, "/register/email", {"email": "user@example.com", "password": "abcdef"})
        assert rv.status_code == 200

    def test_register_stores_hashed_password(self, client, app, tmp_data_dir):
        """Password must never be stored as plaintext."""
        import app as app_module
        register(client, password="supersecret")
        users = app_module._load_users()
        info = users.get("user@example.com", {})
        stored = info.get("password_hash", "") if isinstance(info, dict) else info
        assert stored != "supersecret"
        assert stored.startswith("scrypt:") or stored.startswith("pbkdf2:") or stored.startswith("$")


# ===========================================================================
# LOGIN
# ===========================================================================

class TestLogin:
    def setup_method(self):
        """Each test registers a fresh user via the client fixture."""

    def test_login_success(self, client):
        register(client)
        client.get("/logout")  # clear session from registration
        rv = login(client)
        assert rv.status_code == 200
        assert rv.get_json().get("ok") is True

    def test_login_sets_session(self, client):
        register(client)
        client.get("/logout")
        login(client)
        with client.session_transaction() as sess:
            assert sess.get("user_email") == "user@example.com"

    def test_login_wrong_password(self, client):
        register(client)
        client.get("/logout")
        rv = post_json(client, "/login/email", {"email": "user@example.com", "password": "wrongpass"})
        assert rv.status_code == 401
        assert "invalid" in rv.get_json().get("error", "").lower()

    def test_login_unknown_email(self, client):
        rv = post_json(client, "/login/email", {"email": "nobody@example.com", "password": "secret123"})
        assert rv.status_code == 401

    def test_login_missing_email(self, client):
        rv = post_json(client, "/login/email", {"password": "secret123"})
        assert rv.status_code == 400

    def test_login_missing_password(self, client):
        rv = post_json(client, "/login/email", {"email": "user@example.com"})
        assert rv.status_code == 400

    def test_login_empty_body(self, client):
        rv = post_json(client, "/login/email", {})
        assert rv.status_code == 400

    def test_login_non_json_body(self, client):
        rv = client.post("/login/email", data="not-json", content_type="text/plain")
        assert rv.status_code == 400

    def test_login_email_case_insensitive(self, client):
        """Login with UPPER@example.com should succeed for a lower-registered account."""
        register(client, email="user@example.com")
        client.get("/logout")
        rv = post_json(client, "/login/email", {"email": "USER@EXAMPLE.COM", "password": "secret123"})
        assert rv.status_code == 200


# ===========================================================================
# LOGOUT
# ===========================================================================

class TestLogout:
    def test_logout_clears_session(self, client):
        register(client)
        with client.session_transaction() as sess:
            assert "user_email" in sess
        client.get("/logout")
        with client.session_transaction() as sess:
            assert "user_email" not in sess

    def test_logout_redirects_to_home(self, client):
        register(client)
        rv = client.get("/logout")
        assert rv.status_code in (301, 302)
        assert rv.headers.get("Location", "").endswith("/") or rv.location == "/"

    def test_logout_unauthenticated_is_safe(self, client):
        """Logging out when not logged in must not crash."""
        rv = client.get("/logout")
        assert rv.status_code in (200, 301, 302)


# ===========================================================================
# SESSION PERSISTENCE
# ===========================================================================

class TestSessionPersistence:
    def test_session_survives_multiple_requests(self, client):
        register(client)
        # Make several unrelated requests
        client.get("/")
        client.get("/api/remarkables")
        with client.session_transaction() as sess:
            assert sess.get("user_email") == "user@example.com"

    def test_session_not_shared_between_clients(self, app):
        """Two independent clients must have independent sessions."""
        c1 = app.test_client()
        c2 = app.test_client()
        post_json(c1, "/register/email", {"email": "a@example.com", "password": "secret123"})
        post_json(c2, "/register/email", {"email": "b@example.com", "password": "secret123"})
        with c1.session_transaction() as s1:
            assert s1.get("user_email") == "a@example.com"
        with c2.session_transaction() as s2:
            assert s2.get("user_email") == "b@example.com"


# ===========================================================================
# WATCHLIST AUTH GATE
# ===========================================================================

class TestWatchlistAuth:
    def test_watchlist_get_requires_auth(self, client):
        rv = client.get("/api/watchlist")
        assert rv.status_code == 401

    def test_watchlist_post_requires_auth(self, client):
        rv = post_json(client, "/api/watchlist", {"symbols": ["AAPL"]})
        assert rv.status_code == 401

    def test_watchlist_requires_pro(self, client):
        """Free-tier users must be denied watchlist access with 403."""
        register(client)
        rv = client.get("/api/watchlist")
        assert rv.status_code == 403
        assert rv.get_json().get("required_tier") == "pro"

    def test_watchlist_accessible_for_pro(self, client, monkeypatch):
        import app as app_module
        register(client)
        monkeypatch.setattr(app_module, "_get_user_tier", lambda e: "pro")
        rv = client.get("/api/watchlist")
        assert rv.status_code == 200
        assert "symbols" in rv.get_json()

    def test_watchlist_save_and_retrieve_pro(self, client, monkeypatch):
        import app as app_module
        register(client)
        monkeypatch.setattr(app_module, "_get_user_tier", lambda e: "pro")
        post_json(client, "/api/watchlist", {"symbols": ["AAPL", "TSLA"]})
        rv = client.get("/api/watchlist")
        assert rv.status_code == 200
        assert set(rv.get_json()["symbols"]) == {"AAPL", "TSLA"}

    def test_watchlist_cleared_after_logout(self, client):
        register(client)
        client.get("/logout")
        rv = client.get("/api/watchlist")
        assert rv.status_code == 401


# ===========================================================================
# GOOGLE OAUTH (mocked)
# ===========================================================================

class TestGoogleOAuth:
    def test_google_oauth_not_configured_when_creds_missing(self, app):
        """If GOOGLE_OAUTH_CLIENT_ID/SECRET are empty, google_bp must be None."""
        import app as app_module
        # Env vars were blanked in module-level setup
        assert app_module.google_bp is None

    def test_google_oauth_sets_session_on_authorized(self, client, monkeypatch):
        """
        Simulate a Google OAuth callback: monkeypatch both user email and tier
        so the watchlist endpoint accepts the request.
        """
        import app as app_module
        monkeypatch.setattr(app_module, "_get_current_user_email", lambda: "guser@gmail.com")
        monkeypatch.setattr(app_module, "_get_user_tier", lambda e: "pro")
        rv = client.get("/api/watchlist")
        assert rv.status_code == 200

    def test_google_oauth_no_session_without_token(self, client):
        """Without a Google token or session, _get_current_user_email returns None."""
        import app as app_module
        # google_bp is None (no creds), so the google.authorized branch is skipped
        with client.application.test_request_context():
            email = app_module._get_current_user_email()
        assert email is None


# ===========================================================================
# TOKEN / SESSION EXPIRY
# ===========================================================================

class TestSessionExpiry:
    def test_expired_session_denies_watchlist(self, client):
        """Manually clear the session to simulate expiry; watchlist must return 401."""
        register(client)
        # Forcibly wipe the session
        with client.session_transaction() as sess:
            sess.clear()
        rv = client.get("/api/watchlist")
        assert rv.status_code == 401

    def test_re_login_after_session_expiry(self, client, monkeypatch):
        """User can re-authenticate after session is cleared."""
        import app as app_module
        register(client)
        with client.session_transaction() as sess:
            sess.clear()
        login(client)
        monkeypatch.setattr(app_module, "_get_user_tier", lambda e: "pro")
        rv = client.get("/api/watchlist")
        assert rv.status_code == 200
