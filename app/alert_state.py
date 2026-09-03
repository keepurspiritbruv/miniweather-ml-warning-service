from datetime import datetime, timezone
import json
from pathlib import Path

LEVEL_RANK = {"NORMAL": 0, "WASPADA": 1, "SIAGA": 2, "AWAS": 3}


class AlertCooldown:
    def __init__(self, path: Path):
        self.path = Path(path)

    def should_post(
        self,
        hazard: str,
        level: str,
        now: datetime,
        cooldown_hours: float,
    ) -> tuple[bool, str]:
        state = self._load()
        last = state.get(hazard)

        if level == "NORMAL" or level not in LEVEL_RANK:
            return False, "no_post_level"

        if last is None:
            state[hazard] = {"level": level, "ts": now.timestamp()}
            self._save(state)
            return True, "first_post"

        if LEVEL_RANK[level] > LEVEL_RANK[last["level"]]:
            state[hazard] = {"level": level, "ts": now.timestamp()}
            self._save(state)
            return True, "escalation"

        if LEVEL_RANK[level] < LEVEL_RANK[last["level"]]:
            return False, "downgrade_suppressed"

        last_posted = datetime.fromtimestamp(last["ts"], tz=timezone.utc)
        elapsed_hours = (now - last_posted).total_seconds() / 3600.0
        if elapsed_hours >= cooldown_hours:
            state[hazard] = {"level": level, "ts": now.timestamp()}
            self._save(state)
            return True, "cooldown_elapsed"

        return False, f"cooldown_active_{elapsed_hours:.1f}h"

    def note_normal(self, hazard: str) -> bool:
        """Forget stored cooldown state when the hazard returns to NORMAL.

        Returns True when a stored record was removed.
        """
        state = self._load()
        if hazard not in state:
            return False
        del state[hazard]
        self._save(state)
        return True

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        state = {}
        for hazard, record in raw.items():
            if not isinstance(record, dict):
                continue
            level = record.get("level")
            try:
                ts = float(record.get("ts"))
            except (TypeError, ValueError):
                continue
            if level in LEVEL_RANK:
                state[hazard] = {"level": level, "ts": ts}
        return state

    def _save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2), encoding="utf-8")
