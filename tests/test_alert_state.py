from datetime import datetime, timezone
import json

import pytest

from app.alert_state import AlertCooldown

NOW = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)


def test_first_post_is_allowed(tmp_path):
    cooldown = AlertCooldown(tmp_path / "state.json")

    allowed, reason = cooldown.should_post("curah_hujan_tinggi", "WASPADA", NOW, 12.0)

    assert allowed is True
    assert reason == "first_post"


def test_same_level_within_cooldown_is_blocked(tmp_path):
    cooldown = AlertCooldown(tmp_path / "state.json")
    cooldown.should_post("curah_hujan_tinggi", "WASPADA", NOW, 12.0)

    later = datetime(2026, 9, 3, 14, 0, 0, tzinfo=timezone.utc)
    allowed, reason = cooldown.should_post("curah_hujan_tinggi", "WASPADA", later, 12.0)

    assert allowed is False
    assert reason.startswith("cooldown_active")


def test_same_level_after_cooldown_is_allowed(tmp_path):
    cooldown = AlertCooldown(tmp_path / "state.json")
    cooldown.should_post("curah_hujan_tinggi", "WASPADA", NOW, 12.0)

    later = datetime(2026, 9, 4, 1, 0, 0, tzinfo=timezone.utc)  # +13h
    allowed, reason = cooldown.should_post("curah_hujan_tinggi", "WASPADA", later, 12.0)

    assert allowed is True
    assert reason == "cooldown_elapsed"


def test_escalation_is_allowed_within_cooldown(tmp_path):
    cooldown = AlertCooldown(tmp_path / "state.json")
    cooldown.should_post("curah_hujan_tinggi", "WASPADA", NOW, 12.0)

    later = datetime(2026, 9, 3, 13, 0, 0, tzinfo=timezone.utc)  # +1h
    allowed, reason = cooldown.should_post("curah_hujan_tinggi", "AWAS", later, 12.0)

    assert allowed is True
    assert reason == "escalation"


def test_downgrade_is_never_posted(tmp_path):
    cooldown = AlertCooldown(tmp_path / "state.json")
    cooldown.should_post("curah_hujan_tinggi", "AWAS", NOW, 12.0)

    later = datetime(2026, 9, 4, 1, 0, 0, tzinfo=timezone.utc)
    allowed, reason = cooldown.should_post("curah_hujan_tinggi", "SIAGA", later, 12.0)

    assert allowed is False
    assert reason == "downgrade_suppressed"


def test_note_normal_removes_stored_state(tmp_path):
    cooldown = AlertCooldown(tmp_path / "state.json")
    cooldown.should_post("curah_hujan_tinggi", "AWAS", NOW, 12.0)

    assert cooldown.note_normal("curah_hujan_tinggi") is True

    later = datetime(2026, 9, 4, 1, 0, 0, tzinfo=timezone.utc)
    allowed, reason = cooldown.should_post("curah_hujan_tinggi", "WASPADA", later, 12.0)

    assert allowed is True
    assert reason == "first_post"


def test_note_normal_without_state_returns_false(tmp_path):
    cooldown = AlertCooldown(tmp_path / "state.json")

    assert cooldown.note_normal("curah_hujan_tinggi") is False
    assert not (tmp_path / "state.json").exists()


def test_hazards_are_isolated(tmp_path):
    cooldown = AlertCooldown(tmp_path / "state.json")
    cooldown.should_post("curah_hujan_tinggi", "WASPADA", NOW, 12.0)

    later = datetime(2026, 9, 3, 13, 0, 0, tzinfo=timezone.utc)
    allowed, _ = cooldown.should_post("angin_kencang", "WASPADA", later, 12.0)

    assert allowed is True


def test_corrupt_state_file_is_ignored(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("{", encoding="utf-8")
    cooldown = AlertCooldown(state_path)

    allowed, reason = cooldown.should_post("curah_hujan_tinggi", "WASPADA", NOW, 12.0)

    assert allowed is True
    assert reason == "first_post"


def test_non_dict_state_file_is_ignored(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("[]", encoding="utf-8")
    cooldown = AlertCooldown(state_path)

    allowed, reason = cooldown.should_post("curah_hujan_tinggi", "WASPADA", NOW, 12.0)

    assert allowed is True
    assert reason == "first_post"


def test_state_with_unknown_level_is_ignored(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"curah_hujan_tinggi": {"level": "FOO", "ts": NOW.timestamp()}}),
        encoding="utf-8",
    )
    cooldown = AlertCooldown(state_path)

    later = datetime(2026, 9, 3, 13, 0, 0, tzinfo=timezone.utc)
    allowed, reason = cooldown.should_post("curah_hujan_tinggi", "WASPADA", later, 12.0)

    assert allowed is True
    assert reason == "first_post"
