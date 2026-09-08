from dataclasses import dataclass
from datetime import datetime
from typing import Any
import re
import uuid

from app.config import Settings


TABLE_RE = re.compile(r"^records_[a-zA-Z0-9_]+$")


def parse_collection_id(value: str) -> uuid.UUID:
    """Parse a collection id string into a uuid.UUID, failing fast on bad input."""
    return uuid.UUID(str(value).strip())


@dataclass(frozen=True)
class WeatherRow:
    row_id: str
    collection_id: str
    created_by: str | None
    updated_at: datetime
    curah_hujan: float | None
    kecepatan_angin: float | None
    arah_angin: float | None


def build_weather_query(table_name: str) -> str:
    if not TABLE_RE.match(table_name):
        raise ValueError(f"Invalid Scylla table name: {table_name}")

    return f"""
        SELECT "_id", "_collection_id", "_created_by", "_updated_at",
               curah_hujan, kecepatan_angin, arah_angin
        FROM {table_name}
        WHERE "_collection_id" = ? AND "_updated_at" >= ? AND "_updated_at" <= ?
        LIMIT ?
        ALLOW FILTERING
    """


def _to_str(value: Any) -> str:
    return str(value) if value is not None else ""


def _to_float(value: Any) -> float | None:
    return float(value) if value is not None else None


class ScyllaWeatherClient:
    def __init__(self, settings: Settings):
        from cassandra.auth import PlainTextAuthProvider
        from cassandra.cluster import Cluster
        from cassandra.policies import DCAwareRoundRobinPolicy

        self.settings = settings
        self.collection_uuid = parse_collection_id(settings.scylla_collection_id)
        auth_provider = PlainTextAuthProvider(
            username=settings.scylla_username,
            password=settings.scylla_password,
        )
        self.cluster = Cluster(
            contact_points=settings.scylla_contact_points,
            port=settings.scylla_port,
            auth_provider=auth_provider,
            protocol_version=4,
            load_balancing_policy=DCAwareRoundRobinPolicy(local_dc="datacenter1"),
        )
        self.session = self.cluster.connect(settings.scylla_keyspace)
        self.query = build_weather_query(settings.scylla_table)
        self.statement = self.session.prepare(self.query)

    def close(self) -> None:
        self.cluster.shutdown()

    def fetch_rows(self, start_time: datetime, end_time: datetime) -> list[WeatherRow]:
        result = self.session.execute(
            self.statement,
            (
                self.collection_uuid,
                start_time,
                end_time,
                self.settings.query_limit,
            ),
        )
        rows: list[WeatherRow] = []
        for row in result:
            rows.append(
                WeatherRow(
                    row_id=_to_str(row.id),
                    collection_id=_to_str(row.collection_id),
                    created_by=_to_str(row.created_by) or None,
                    updated_at=row.updated_at,
                    curah_hujan=_to_float(row.curah_hujan),
                    kecepatan_angin=_to_float(row.kecepatan_angin),
                    arah_angin=_to_float(row.arah_angin),
                )
            )
        return sorted(rows, key=lambda item: item.updated_at)
