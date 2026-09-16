"""
media_cataloger_client.py
Production HTTP client with JWT session management, RBAC, and Secret Vault support.
Complies with AI_ENGINE_SECURITY_INTEGRATION.md specifications.
"""
import time
import requests
from typing import Optional, Dict, Any, List, Union


class CatalogerApiError(Exception):
    """Base exception for all Media Cataloger API communication errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class AuthenticationError(CatalogerApiError):
    """Raised when authentication fails (HTTP 401) or token is invalid/expired."""
    pass


class PermissionDeniedError(CatalogerApiError):
    """Raised when an operation lacks necessary permissions (HTTP 403)."""
    pass


class VaultLockedError(CatalogerApiError):
    """Raised when accessing private vault resources without an unlocked session."""
    pass


class InvalidRequestError(CatalogerApiError):
    """Raised on invalid request payloads or bad parameters (HTTP 400)."""
    pass


class MediaCatalogerClient:
    """
    Robust API client for interacting with media_cataloger and media_cataloger_web services.
    Supports JWT Bearer authentication, automatic token refresh, RBAC, and Secret Vault token isolation.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        username: Optional[str] = None,
        password: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: float = 15.0
    ):

        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.username = username
        self.password = password
        self.token: Optional[str] = None
        self.vault_token: Optional[str] = None
        self.token_expiry: float = 0
        self.current_user: Optional[Dict[str, Any]] = None

    def _handle_response(self, response: requests.Response) -> Any:
        """Parse JSON response and raise domain-specific exceptions on HTTP errors."""
        try:
            data = response.json()
        except Exception:
            data = {"detail": response.text}

        if response.status_code == 401:
            detail = data.get("detail", "Authentication failed or token expired.") if isinstance(data, dict) else str(data)
            raise AuthenticationError(f"HTTP 401 Unauthorized: {detail}", status_code=401, response_data=data)

        if response.status_code == 403:
            detail = data.get("detail", "Access forbidden. Missing required permissions.") if isinstance(data, dict) else str(data)
            raise PermissionDeniedError(f"HTTP 403 Forbidden: {detail}", status_code=403, response_data=data)

        if response.status_code == 400:
            detail = data.get("detail", "Bad Request") if isinstance(data, dict) else str(data)
            raise InvalidRequestError(f"HTTP 400 Bad Request: {detail}", status_code=400, response_data=data)

        if not response.ok:
            detail = data.get("detail", response.reason) if isinstance(data, dict) else str(data)
            raise CatalogerApiError(f"HTTP {response.status_code} Error: {detail}", status_code=response.status_code, response_data=data)

        return data

    def _request(self, method: str, path: str, **kwargs) -> Any:
        """Internal request wrapper with automatic authentication refresh on 401."""
        url = f"{self.base_url}{path}"
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout

        # Check if token is nearing expiration (< 60s) and credentials are saved
        if self.token and self.username and self.password and self.token_expiry > 0 and (self.token_expiry - time.time() < 60):
            try:
                self.login(self.username, self.password)
            except Exception:
                pass


        try:
            resp = self.session.request(method, url, **kwargs)
            return self._handle_response(resp)
        except AuthenticationError:
            # Attempt automatic re-login if credentials are saved
            if self.username and self.password:
                self.login(self.username, self.password)
                resp = self.session.request(method, url, **kwargs)
                return self._handle_response(resp)
            raise

    # --- Authentication & Session Management ---

    def login(self, username: Optional[str] = None, password: Optional[str] = None) -> Dict[str, Any]:
        """
        Authenticate and acquire JWT session token.
        Saves credentials for automatic renewal if provided.
        """
        user = username or self.username
        pwd = password or self.password
        if not user or not pwd:
            raise AuthenticationError("Username and password are required for login.")

        self.username = user
        self.password = pwd

        url = f"{self.base_url}/api/auth/login"
        resp = self.session.post(url, json={"username": user, "password": pwd}, timeout=self.timeout)
        data = self._handle_response(resp)

        self.token = data["token"]
        self.token_expiry = time.time() + (23 * 3600)  # 23h validity
        self.current_user = data.get("user")
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        return self.current_user

    def set_token(self, token: str, expiry_seconds: float = 82800):
        """Manually configure an existing JWT token."""
        self.token = token
        self.token_expiry = time.time() + expiry_seconds
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    # --- Secret Vault Management ---

    def unlock_vault(self, pin: str = "1234") -> str:
        """
        Unlock Secret Vault session to process private/isolated media files.
        Attaches the `x-vault-token` header to all subsequent requests.
        """
        url = f"{self.base_url}/api/vault/unlock"
        resp = self.session.post(url, json={"pin": pin}, timeout=self.timeout)
        data = self._handle_response(resp)

        self.vault_token = data.get("sessionToken")
        if self.vault_token:
            self.session.headers.update({"x-vault-token": self.vault_token})
        return self.vault_token

    def lock_vault(self) -> None:
        """Terminate and lock the Secret Vault session."""
        try:
            if self.vault_token:
                self.session.post(f"{self.base_url}/api/vault/lock", timeout=5)
        finally:
            self.vault_token = None
            self.session.headers.pop("x-vault-token", None)

    def get_vault_status(self) -> Dict[str, Any]:
        """Check whether the Secret Vault is currently unlocked."""
        return self._request("GET", "/api/vault/status")

    # --- Pipeline Execution Controls (admin_panel) ---

    def trigger_full_sync(
        self,
        force: bool = False,
        files: Optional[List[Any]] = None,
        output_folder: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Trigger whole-archive media processing sync (requires admin_panel permission)."""
        payload: Dict[str, Any] = {"force": force}
        if files is not None:
            payload["files"] = files
            payload["total_files"] = len(files)
        if output_folder:
            payload["output_folder"] = output_folder
        if settings:
            payload["settings"] = settings

        return self._request("POST", f"/api/run?force={str(force).lower()}", json=payload)

    def trigger_single_analysis(
        self,
        file_path: str,
        output_folder: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze a single media file immediately (requires admin_panel permission)."""
        payload: Dict[str, Any] = {"file": file_path}
        if output_folder:
            payload["output_folder"] = output_folder
        if settings:
            payload["settings"] = settings

        return self._request("POST", f"/api/analyze-file?file={file_path}", json=payload)

    def pause(self) -> Dict[str, Any]:
        """Pause active pipeline processing."""
        return self._request("POST", "/api/pause")

    def resume(self) -> Dict[str, Any]:
        """Resume paused pipeline processing."""
        return self._request("POST", "/api/resume")

    def stop(self) -> Dict[str, Any]:
        """Cancel and stop active pipeline processing."""
        return self._request("POST", "/api/stop")

    # --- Metadata Ingestion (edit_metadata) ---

    def save_metadata(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest AI analysis, tags, transcripts, and localized descriptions into database and sidecars.
        Requires edit_metadata permission.
        """
        payload = {"file": file_path, **metadata}
        return self._request("POST", "/api/media/metadata", json=payload)

    # --- Face Recognition & Registry (manage_faces) ---

    def assign_face(
        self,
        file_path: str,
        face_id: str,
        person_name: str,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Register a recognized individual face crop into registry and link to source file.
        Requires manage_faces permission.
        """
        payload = {
            "file": file_path,
            "face_id": face_id,
            "person_name": person_name,
            "confidence": confidence,
        }
        return self._request("POST", "/api/faces/assign", json=payload)

    def rename_face(self, face_id: str, new_name: str) -> Dict[str, Any]:
        """Rename a registered face identity (requires manage_faces)."""
        return self._request("POST", "/api/faces/rename", json={"face_id": face_id, "name": new_name})

    def reset_face(self, face_id: str) -> Dict[str, Any]:
        """Reset a face assignment back to unassigned candidate."""
        return self._request("POST", "/api/faces/reset", json={"face_id": face_id})

    def delete_face(self, face_id: str) -> Dict[str, Any]:
        """Delete a face crop from the registry (requires manage_faces)."""
        return self._request("POST", "/api/faces/delete", json={"face_id": face_id})

    # --- Status, Media & Diagnostics (Public / Read) ---

    def get_status(self) -> Dict[str, Any]:
        """Get live worker pipeline progress and queue status."""
        return self._request("GET", "/api/status")

    def get_queue(self) -> Dict[str, Any]:
        """Get background worker queue status."""
        return self._request("GET", "/api/queue")

    def get_logs(self) -> Dict[str, Any]:
        """Get execution logs."""
        return self._request("GET", "/api/logs")

    def clear_logs(self) -> Dict[str, Any]:
        """Clear execution logs (requires admin_panel)."""
        return self._request("POST", "/api/logs/clear")

    def get_settings(self) -> Dict[str, Any]:
        """Get path and engine configuration."""
        return self._request("GET", "/api/settings")

    def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update persistent path and engine configuration (requires admin_panel)."""
        return self._request("POST", "/api/settings", json=settings)

    def get_media_files(self, vault: bool = False) -> Dict[str, Any]:
        """List all discovered media files with face and metadata summaries."""
        params = {"vault": "true"} if vault else {}
        return self._request("GET", "/api/media/files", params=params)

    def get_faces(self) -> List[Dict[str, Any]]:
        """List all registered face identities in registry."""
        return self._request("GET", "/api/faces")

    def get_sidecar(self, file_path: str) -> Dict[str, Any]:
        """Get raw sidecar JSON metadata for a media file."""
        return self._request("GET", "/api/media/sidecar", params={"file": file_path})

    def health(self) -> Dict[str, Any]:
        """Check API daemon health status."""
        try:
            return self._request("GET", "/api/health")
        except Exception:
            return self._request("GET", "/health")


# --- Example CLI Usage ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Media Cataloger AI Engine Client CLI")
    parser.add_argument("--url", default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--user", default="admin", help="Username")
    parser.add_argument("--password", default="admin", help="Password")
    parser.add_argument("--status", action="store_true", help="Print daemon status")
    args = parser.parse_args()

    client = MediaCatalogerClient(base_url=args.url)
    try:
        user = client.login(args.user, args.password)
        print(f"Logged in as: {user.get('displayName')} ({user.get('role')})")
        status = client.get_status()
        print(f"Status: {status.get('status')}")
    except Exception as e:
        print(f"Connection failed: {e}")
