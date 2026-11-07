"""
Google Drive REST API Client and Authentication Management.
Handles OAuth 2.0 user tokens and Service Account JWT authentication.
"""
import os
import time
import json
import threading
import requests

try:
    import jwt
    import cryptography
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False


class DriveClient:
    """Manages credentials, tokens, and authorized HTTP headers for Google Drive API v3."""

    OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
    DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
    DRIVE_UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"
    SCOPES = ["https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

    def __init__(self, service_account_path=None, root_folder_id=None):
        self.service_account_path = (
            service_account_path
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
        )
        self.root_folder_id = root_folder_id or os.environ.get("GDRIVE_ROOT_FOLDER_ID") or "root"
        self.access_token = os.environ.get("GDRIVE_ACCESS_TOKEN")
        self.refresh_token = os.environ.get("GDRIVE_REFRESH_TOKEN")
        self.client_id = os.environ.get("GDRIVE_CLIENT_ID")
        self.client_secret = os.environ.get("GDRIVE_CLIENT_SECRET")
        self.token_file_path = os.environ.get("GDRIVE_TOKEN_JSON")
        
        self.token_expiry = 0
        self.folder_cache = {}  # Cache: (parent_id, folder_name) -> folder_id
        self._lock = threading.Lock()

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if not self.token_file_path:
            cand_token = os.path.join(base_dir, "token.json")
            if os.path.exists(cand_token):
                self.token_file_path = cand_token

        # Check for service account / OAuth credentials file in standard locations
        if not self.service_account_path:
            for candidate in ("service_account.json", "credentials.json", "gdrive_credentials.json"):
                cand_path = os.path.join(base_dir, candidate)
                if os.path.exists(cand_path):
                    self.service_account_path = cand_path
                    break

        # Check if the file is an OAuth Client credentials file
        if self.service_account_path and os.path.exists(self.service_account_path):
            try:
                with open(self.service_account_path, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                client_info = cdata.get("installed") or cdata.get("web")
                if client_info:
                    self.client_id = self.client_id or client_info.get("client_id")
                    self.client_secret = self.client_secret or client_info.get("client_secret")
            except Exception as e:
                print(f"[gdrive] Warning inspecting credentials file: {e}")

        # Load token from token.json if present
        if self.token_file_path and os.path.exists(self.token_file_path):
            try:
                with open(self.token_file_path, "r", encoding="utf-8") as f:
                    tdata = json.load(f)
                self.access_token = self.access_token or tdata.get("access_token")
                self.refresh_token = self.refresh_token or tdata.get("refresh_token")
                self.token_expiry = tdata.get("expiry", 0)
                if not self.client_id:
                    self.client_id = tdata.get("client_id")
                if not self.client_secret:
                    self.client_secret = tdata.get("client_secret")
            except Exception as e:
                print(f"[gdrive] Warning reading token.json: {e}")

    def is_configured(self):
        """Returns True if valid Google Drive credentials or access tokens are available."""
        if self.refresh_token and self.client_id and self.client_secret:
            return True
        if self.access_token:
            return True
        if self.service_account_path and os.path.exists(self.service_account_path):
            try:
                with open(self.service_account_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("type") == "service_account":
                    return True
            except Exception:
                pass
        return False

    def _get_auth_headers(self):
        """Returns HTTP Authorization headers with a valid, refreshed access token."""
        token = self._get_access_token()
        if not token:
            raise RuntimeError("Google Drive credentials not configured or token generation failed.")
        return {
            "Authorization": f"Bearer {token}",
        }

    def _get_access_token(self):
        """Refreshes or generates access token using Service Account JWT or OAuth2 refresh flow."""
        with self._lock:
            now = time.time()
            if self.access_token and now < (self.token_expiry - 60):
                return self.access_token

            # Option 1: Service Account JSON with JWT
            if self.service_account_path and os.path.exists(self.service_account_path) and _JWT_AVAILABLE:
                try:
                    with open(self.service_account_path, "r", encoding="utf-8") as f:
                        sa_data = json.load(f)
                    
                    if sa_data.get("type") == "service_account":
                        private_key = sa_data.get("private_key")
                        client_email = sa_data.get("client_email")
                        token_uri = sa_data.get("token_uri", self.OAUTH_TOKEN_URL)

                        if private_key and client_email:
                            payload = {
                                "iss": client_email,
                                "scope": " ".join(self.SCOPES),
                                "aud": token_uri,
                                "exp": int(now + 3600),
                                "iat": int(now),
                            }
                            signed_jwt = jwt.encode(payload, private_key, algorithm="RS256")
                            
                            resp = requests.post(
                                token_uri,
                                data={
                                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                                    "assertion": signed_jwt,
                                },
                                timeout=15,
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                self.access_token = data.get("access_token")
                                self.token_expiry = now + data.get("expires_in", 3600)
                                return self.access_token
                            else:
                                print(f"[gdrive] Service Account token exchange failed: {resp.status_code} {resp.text}")
                except Exception as e:
                    print(f"[gdrive] Error generating Service Account token: {e}")

            # Option 2: OAuth2 Refresh Token (from credentials.json / token.json)
            if self.refresh_token and self.client_id and self.client_secret:
                try:
                    resp = requests.post(
                        self.OAUTH_TOKEN_URL,
                        data={
                            "client_id": self.client_id,
                            "client_secret": self.client_secret,
                            "refresh_token": self.refresh_token,
                            "grant_type": "refresh_token",
                        },
                        timeout=15,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        self.access_token = data.get("access_token")
                        self.token_expiry = now + data.get("expires_in", 3600)
                        if self.token_file_path:
                            try:
                                tdata = {}
                                if os.path.exists(self.token_file_path):
                                    with open(self.token_file_path, "r", encoding="utf-8") as tf:
                                        tdata = json.load(tf)
                                tdata["access_token"] = self.access_token
                                tdata["expiry"] = self.token_expiry
                                tdata["refresh_token"] = self.refresh_token
                                tdata["client_id"] = self.client_id
                                tdata["client_secret"] = self.client_secret
                                with open(self.token_file_path, "w", encoding="utf-8") as tf:
                                    json.dump(tdata, tf, indent=2)
                            except Exception:
                                pass
                        return self.access_token
                    else:
                        print(f"[gdrive] OAuth2 refresh failed: {resp.status_code} {resp.text}")
                except Exception as e:
                    print(f"[gdrive] Error refreshing OAuth token: {e}")

            return self.access_token

