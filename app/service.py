from datetime import datetime, timedelta, timezone
import logging

from app.alert_rules import AlertDecision, decide_alert
from app.config import Settings
from app.features import build_daily_features

LOGGER = logging.getLogger(__name__)


def _latest_values(daily_df, hazard: str) -> dict[str, float]:
    if daily_df.empty:
        return {}
    latest = daily_df.iloc[-1]
    if hazard == "curah_hujan_tinggi":
        return {"RR": float(latest.get("RR", 0.0))}
    if hazard == "angin_kencang":
        return {
            "ff_x": float(latest.get("ff_x", 0.0)),
            "ff_avg": float(latest.get("ff_avg", 0.0)),
        }
    return {}


def _is_stale(daily_df, stale_after_hours: int) -> bool:
    if daily_df.empty or "date" not in daily_df.columns:
        return False
    latest_date = datetime.combine(daily_df.iloc[-1]["date"], datetime.min.time()).replace(
        tzinfo=timezone.utc
    )
    return datetime.now(timezone.utc) - latest_date > timedelta(hours=stale_after_hours)


def run_once(settings: Settings, weather_client, backend_client, runners) -> list[AlertDecision]:
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=35)
    rows = weather_client.fetch_rows(start_time=start_time, end_time=end_time)
    daily_df = build_daily_features(rows, timezone_name=settings.timezone)
    stale = _is_stale(daily_df, settings.stale_after_hours)

    decisions: list[AlertDecision] = []
    for runner in runners:
        result = runner.score_latest(daily_df)
        decision = decide_alert(result, _latest_values(daily_df, result.hazard), settings)
        if stale and decision.status not in {"WAITING_FOR_HISTORY", "ERROR"}:
            decision = AlertDecision(
                hazard=decision.hazard,
                status="WAITING_FOR_RECENT_HISTORY",
                should_post=False,
                message=f"{decision.hazard}: latest Scylla data is older than configured limit",
                score=decision.score,
                reason="Latest data is stale",
            )
        decisions.append(decision)

        LOGGER.info(
            "hazard=%s status=%s score=%s dry_run=%s reason=%s message=%s",
            decision.hazard,
            decision.status,
            decision.score,
            settings.dry_run,
            decision.reason,
            decision.message,
        )

        if decision.should_post and not settings.dry_run:
            backend_client.create_warning(decision.message)

    return decisions
