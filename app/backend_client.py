from typing import Any

import requests

from app.config import Settings


class MiniweatherBackendClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or requests.Session()
        self.access_token: str | None = None
        self.refresh_token: str | None = None

    def login(self) -> None:
        response = self.session.post(
            f"{self.settings.backend_base_url}/v1/auth/login",
            json={
                "email": self.settings.miniweather_auth_email,
                "password": self.settings.miniweather_auth_password,
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()["data"]
        self.access_token = data["accessToken"]
        self.refresh_token = data["refreshToken"]

    def create_warning(
        self,
        message: str,
        warning_type: str = "weather",
        is_active: bool = True,
    ) -> dict[str, Any]:
        if not self.access_token:
            self.login()

        response = self.session.post(
            f"{self.settings.backend_base_url}/v1/warnings",
            json={
                "message": message,
                "type": warning_type,
                "is_active": is_active,
            },
            headers={"Authorization": f"Bearer {self.access_token}"},
            timeout=20,
        )

        if response.status_code == 401:
            self.login()
            response = self.session.post(
                f"{self.settings.backend_base_url}/v1/warnings",
                json={
                    "message": message,
                    "type": warning_type,
                    "is_active": is_active,
                },
                headers={"Authorization": f"Bearer {self.access_token}"},
                timeout=20,
            )

        response.raise_for_status()
        return response.json()
