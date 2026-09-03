from app.config import load_settings


def test_load_settings_defaults():
    settings = load_settings(
        {
            "SCYLLA_PASSWORD": "secret",
            "MINIWEATHER_AUTH_EMAIL": "admin@example.com",
            "MINIWEATHER_AUTH_PASSWORD": "admin-secret",
        }
    )

    assert settings.data_source == "scylla"
    assert settings.scylla_contact_points == ["10.42.28.70"]
    assert settings.scylla_port == 9042
    assert settings.scylla_keyspace == "hyperbase"
    assert settings.scylla_username == "hyperbase"
    assert settings.scylla_password == "secret"
    assert settings.scylla_table == "records_91e2e000cb17494c8a45c6182b2a89ac"
    assert settings.scylla_collection_id == "91e2e000-cb17-494c-8a45-c6182b2a89ac"
    assert settings.backend_base_url == "http://miniweather-backend:3001"
    assert settings.run_interval_minutes == 30
    assert settings.timezone == "Asia/Jakarta"
    assert settings.dry_run is True
    assert settings.skip_partial_today is True
    assert settings.rain_siaga_mm == 50.0
    assert settings.rain_awas_mm == 100.0
    assert settings.wind_siaga_ms == 10.8
    assert settings.wind_awas_ms == 17.2


def test_load_settings_boolean_false():
    settings = load_settings(
        {
            "SCYLLA_PASSWORD": "secret",
            "MINIWEATHER_AUTH_EMAIL": "admin@example.com",
            "MINIWEATHER_AUTH_PASSWORD": "admin-secret",
            "DRY_RUN": "false",
        }
    )

    assert settings.dry_run is False


def test_load_settings_multiple_contact_points():
    settings = load_settings(
        {
            "SCYLLA_CONTACT_POINTS": "10.42.28.70,10.42.28.71",
            "SCYLLA_PASSWORD": "secret",
            "MINIWEATHER_AUTH_EMAIL": "admin@example.com",
            "MINIWEATHER_AUTH_PASSWORD": "admin-secret",
        }
    )

    assert settings.scylla_contact_points == ["10.42.28.70", "10.42.28.71"]
