from app.alert_rules import decide_alert
from app.config import load_settings
from app.model_runner import ModelResult


def settings():
    return load_settings(
        {
            "SCYLLA_PASSWORD": "secret",
            "MINIWEATHER_AUTH_EMAIL": "admin@example.com",
            "MINIWEATHER_AUTH_PASSWORD": "admin-secret",
        }
    )


def result(hazard: str, score: float | None) -> ModelResult:
    return ModelResult(
        experiment_id="x",
        hazard=hazard,
        status="SCORED" if score is not None else "WAITING_FOR_HISTORY",
        score=score,
        thresholds={"p95": 1.0, "p99": 2.0, "p995": 3.0},
        reason="OK",
    )


def test_rain_waspada_uses_score_only():
    decision = decide_alert(result("curah_hujan_tinggi", 1.5), {"RR": 10.0}, settings())

    assert decision.status == "WASPADA"
    assert decision.should_post is True


def test_rain_siaga_requires_physical_threshold():
    decision = decide_alert(result("curah_hujan_tinggi", 2.5), {"RR": 10.0}, settings())

    assert decision.status == "WASPADA"

    decision = decide_alert(result("curah_hujan_tinggi", 2.5), {"RR": 60.0}, settings())

    assert decision.status == "SIAGA"


def test_wind_awas_requires_score_and_physical_threshold():
    decision = decide_alert(result("angin_kencang", 4.0), {"ff_x": 5.0}, settings())

    assert decision.status == "WASPADA"

    decision = decide_alert(result("angin_kencang", 4.0), {"ff_x": 20.0}, settings())

    assert decision.status == "AWAS"


def test_waiting_history_does_not_post():
    decision = decide_alert(result("angin_kencang", None), {"ff_x": 0.0}, settings())

    assert decision.status == "WAITING_FOR_HISTORY"
    assert decision.should_post is False
