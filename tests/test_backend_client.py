from app.backend_client import MiniweatherBackendClient
from app.config import load_settings


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.posts = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if url.endswith("/v1/auth/login"):
            return FakeResponse(
                {
                    "data": {
                        "accessToken": "access-token",
                        "refreshToken": "refresh-token",
                    }
                }
            )
        return FakeResponse({"data": {"warning": {"id": "warning-id"}}}, status_code=201)


def settings():
    return load_settings(
        {
            "SCYLLA_PASSWORD": "secret",
            "MINIWEATHER_AUTH_EMAIL": "admin@example.com",
            "MINIWEATHER_AUTH_PASSWORD": "admin-secret",
        }
    )


def test_login_and_create_warning():
    session = FakeSession()
    client = MiniweatherBackendClient(settings(), session=session)

    created = client.create_warning("test warning")

    assert created["data"]["warning"]["id"] == "warning-id"
    assert session.posts[0]["url"] == "http://miniweather-backend:3001/v1/auth/login"
    assert session.posts[0]["json"] == {
        "email": "admin@example.com",
        "password": "admin-secret",
    }
    assert session.posts[1]["url"] == "http://miniweather-backend:3001/v1/warnings"
    assert session.posts[1]["json"] == {
        "message": "test warning",
        "type": "weather",
        "is_active": True,
    }
    assert session.posts[1]["headers"] == {"Authorization": "Bearer access-token"}
