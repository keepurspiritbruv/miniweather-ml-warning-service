import pytest

from app.scylla_client import build_weather_query


def test_build_weather_query_uses_collection_timestamp_and_allow_filtering():
    query = build_weather_query("records_91e2e000cb17494c8a45c6182b2a89ac")

    assert 'SELECT "_id", "_collection_id", "_created_by", "_updated_at"' in query
    assert "curah_hujan" in query
    assert "kecepatan_angin" in query
    assert "arah_angin" in query
    assert 'WHERE "_collection_id" = ? AND "_updated_at" >= ? AND "_updated_at" <= ?' in query
    assert "ALLOW FILTERING" in query


def test_build_weather_query_rejects_invalid_table_name():
    with pytest.raises(ValueError):
        build_weather_query("DEPLOY TNTF UGM")
