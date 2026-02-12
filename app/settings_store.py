from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path(".sim_ui_settings.json")


LLM_SETTING_KEYS: list[str] = [
    "llm_provider",
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "openai_base_url",
    "llm_temperature",
    "llm_auto_fetch_models",
    "llm_verify_ssl",
    "llm_ca_bundle_path",
    "llm_model_selected",
    "llm_model_name",
    "llm_remember_settings",
]


def load_ui_settings(path: Path | None = None) -> dict[str, Any]:
    target = path or SETTINGS_PATH
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_ui_settings(settings: dict[str, Any], path: Path | None = None) -> Path:
    target = path or SETTINGS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(settings, indent=2, ensure_ascii=True), encoding="utf-8")

    # Best-effort permission hardening on POSIX systems.
    try:
        os.chmod(target, 0o600)
    except Exception:
        pass

    return target


def clear_ui_settings(path: Path | None = None) -> None:
    target = path or SETTINGS_PATH
    if target.exists():
        target.unlink()


def extract_llm_settings(session_state: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key in LLM_SETTING_KEYS:
        data[key] = session_state.get(key)
    return data
