from dataclasses import dataclass

from app.config import Settings
from app.model_runner import ModelResult


@dataclass(frozen=True)
class AlertDecision:
    hazard: str
    status: str
    should_post: bool
    message: str
    score: float | None
    reason: str


def decide_alert(
    result: ModelResult,
    latest_values: dict[str, float],
    settings: Settings,
) -> AlertDecision:
    if result.status != "SCORED" or result.score is None:
        return AlertDecision(
            hazard=result.hazard,
            status=result.status,
            should_post=False,
            message=f"{result.hazard}: {result.status} - {result.reason}",
            score=result.score,
            reason=result.reason,
        )

    p95 = result.thresholds["p95"]
    p99 = result.thresholds["p99"]
    p995 = result.thresholds["p995"]

    status = "NORMAL"
    if result.score >= p95:
        status = "WASPADA"

    if result.hazard == "curah_hujan_tinggi":
        rainfall = float(latest_values.get("RR", 0.0))
        if result.score >= p995 and rainfall >= settings.rain_awas_mm:
            status = "AWAS"
        elif result.score >= p99 and rainfall >= settings.rain_siaga_mm:
            status = "SIAGA"
        message = (
            f"ML {status} curah hujan tinggi. "
            f"Anomaly score={result.score:.3f}, curah hujan harian={rainfall:.2f} mm."
        )
    elif result.hazard == "angin_kencang":
        wind_speed = float(latest_values.get("ff_x", latest_values.get("ff_avg", 0.0)))
        if result.score >= p995 and wind_speed >= settings.wind_awas_ms:
            status = "AWAS"
        elif result.score >= p99 and wind_speed >= settings.wind_siaga_ms:
            status = "SIAGA"
        message = (
            f"ML {status} angin kencang. "
            f"Anomaly score={result.score:.3f}, kecepatan angin maksimum={wind_speed:.2f} m/s."
        )
    else:
        message = f"ML {status} {result.hazard}. Anomaly score={result.score:.3f}."

    return AlertDecision(
        hazard=result.hazard,
        status=status,
        should_post=status != "NORMAL",
        message=message,
        score=result.score,
        reason="OK",
    )
