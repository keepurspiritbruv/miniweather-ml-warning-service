from datetime import datetime, timedelta, timezone

from app.alert_rules import AlertDecision
from app.config import load_settings
from app.model_runner import ModelResult
from app.scylla_client import WeatherRow
from app.service import run_once


class FakeWeatherClient:
    def fetch_rows(self, start_time, end_time):
        rows = []
        for i in range(30):
            rows.append(
                WeatherRow(
                    row_id=str(i),
                    collection_id="91e2e000-cb17-494c-8a45-c6182b2a89ac",
                    created_by=None,
                    updated_at=datetime.now(timezone.utc) - timedelta(days=29 - i),
                    curah_hujan=0.0,
                    kecepatan_angin=0.0,
                    arah_angin=2.0,
                )
            )
        return rows


class FakeRunner:
    def __init__(self, hazard):
        self.hazard = hazard

    def score_latest(self, daily_df):
        return ModelResult(
            experiment_id="fake",
            hazard=self.hazard,
            status="SCORED",
            score=1.5,
            thresholds={"p95": 1.0, "p99": 2.0, "p995": 3.0},
            reason="OK",
        )


class FakeBackendClient:
    def __init__(self):
        self.created = []

    def create_warning(self, message):
        self.created.append(message)
        return {"ok": True}


def test_run_once_dry_run_does_not_post_warning():
    settings = load_settings(
        {
            "SCYLLA_PASSWORD": "secret",
            "MINIWEATHER_AUTH_EMAIL": "admin@example.com",
            "MINIWEATHER_AUTH_PASSWORD": "admin-secret",
            "DRY_RUN": "true",
        }
    )
    backend = FakeBackendClient()

    decisions = run_once(
        settings=settings,
        weather_client=FakeWeatherClient(),
        backend_client=backend,
        runners=[FakeRunner("curah_hujan_tinggi")],
    )

    assert len(decisions) == 1
    assert isinstance(decisions[0], AlertDecision)
    assert decisions[0].status == "WASPADA"
    assert backend.created == []
