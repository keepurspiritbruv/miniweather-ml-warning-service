from dataclasses import dataclass
from os import environ
from typing import Mapping


def _get(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key, default)
    return value.strip() if isinstance(value, str) else default


def _get_required(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _get_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(env: Mapping[str, str], key: str, default: float) -> float:
    return float(_get(env, key, str(default)))


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    return int(_get(env, key, str(default)))


@dataclass(frozen=True)
class Settings:
    data_source: str
    scylla_contact_points: list[str]
    scylla_port: int
    scylla_keyspace: str
    scylla_username: str
    scylla_password: str
    scylla_table: str
    scylla_collection_id: str
    backend_base_url: str
    miniweather_auth_email: str
    miniweather_auth_password: str
    run_interval_minutes: int
    timezone: str
    dry_run: bool
    skip_partial_today: bool
    rain_siaga_mm: float
    rain_awas_mm: float
    wind_siaga_ms: float
    wind_awas_ms: float
    stale_after_hours: int
    query_limit: int
    alert_cooldown_hours: float
    alert_state_file: str


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    source = environ if env is None else env
    contact_points = [
        item.strip()
        for item in _get(source, "SCYLLA_CONTACT_POINTS", "10.42.28.70").split(",")
        if item.strip()
    ]

    return Settings(
        data_source=_get(source, "DATA_SOURCE", "scylla"),
        scylla_contact_points=contact_points,
        scylla_port=_get_int(source, "SCYLLA_PORT", 9042),
        scylla_keyspace=_get(source, "SCYLLA_KEYSPACE", "hyperbase"),
        scylla_username=_get(source, "SCYLLA_USERNAME", "hyperbase"),
        scylla_password=_get_required(source, "SCYLLA_PASSWORD"),
        scylla_table=_get(
            source,
            "SCYLLA_TABLE",
            "records_91e2e000cb17494c8a45c6182b2a89ac",
        ),
        scylla_collection_id=_get(
            source,
            "SCYLLA_COLLECTION_ID",
            "91e2e000-cb17-494c-8a45-c6182b2a89ac",
        ),
        backend_base_url=_get(
            source,
            "MINIWEATHER_API_BASE_URL",
            "http://miniweather-backend:3001",
        ).rstrip("/"),
        miniweather_auth_email=_get_required(source, "MINIWEATHER_AUTH_EMAIL"),
        miniweather_auth_password=_get_required(source, "MINIWEATHER_AUTH_PASSWORD"),
        run_interval_minutes=_get_int(source, "RUN_INTERVAL_MINUTES", 30),
        timezone=_get(source, "TIMEZONE", "Asia/Jakarta"),
        dry_run=_get_bool(source, "DRY_RUN", True),
        skip_partial_today=_get_bool(source, "SKIP_PARTIAL_TODAY", True),
        rain_siaga_mm=_get_float(source, "RAIN_SIAGA_MM", 50.0),
        rain_awas_mm=_get_float(source, "RAIN_AWAS_MM", 100.0),
        wind_siaga_ms=_get_float(source, "WIND_SIAGA_MS", 10.8),
        wind_awas_ms=_get_float(source, "WIND_AWAS_MS", 17.2),
        stale_after_hours=_get_int(source, "STALE_AFTER_HOURS", 48),
        query_limit=_get_int(source, "SCYLLA_QUERY_LIMIT", 250000),
        alert_cooldown_hours=_get_float(source, "ALERT_COOLDOWN_HOURS", 12.0),
        alert_state_file=_get(source, "ALERT_STATE_FILE", "alert_state.json"),
    )
