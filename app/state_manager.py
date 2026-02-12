from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.default_state import STAGE_NAMES, TRAIT_NAMES, build_relationships, create_default_state, pair_key


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    data = state or create_default_state()

    data.setdefault("scenario", {})
    scenario = data["scenario"]
    scenario.setdefault("name", "Untitled Scenario")
    scenario.setdefault("start_stage", 1)
    scenario.setdefault("environment_description", "")
    scenario.setdefault("style_notes", "")
    scenario.setdefault("style_examples", [])
    scenario.setdefault("safe_word", "")
    scenario.setdefault("flags", {})
    scenario.setdefault("initial_tension", float(data.get("tension", 25.0)))
    scenario.setdefault("initial_risk", float(data.get("risk", 15.0)))
    scenario["style_notes"] = str(scenario.get("style_notes", "")).strip()
    raw_style_examples = scenario.get("style_examples", [])
    if not isinstance(raw_style_examples, list):
        raw_style_examples = []
    scenario["style_examples"] = [str(item).strip() for item in raw_style_examples if str(item).strip()][:20]

    data.setdefault("characters", [])
    for idx, char in enumerate(data["characters"]):
        char.setdefault("id", f"char_{idx}")
        char.setdefault("name", f"Character {idx + 1}")
        char.setdefault("role", "npc")
        char.setdefault("description", "")
        char.setdefault("traits", {})
        for trait_name in TRAIT_NAMES:
            trait = char["traits"].setdefault(
                trait_name,
                {
                    "base": 50,
                    "current": 50.0,
                    "volatility": 0.35,
                    "dynamic_enabled": True,
                },
            )
            trait["base"] = int(clamp(float(trait.get("base", 50)), 0, 100))
            trait["current"] = float(clamp(float(trait.get("current", trait["base"])), 0, 100))
            trait["volatility"] = float(clamp(float(trait.get("volatility", 0.35)), 0, 1))
            trait["dynamic_enabled"] = bool(trait.get("dynamic_enabled", True))

    data.setdefault("relationships", {})
    if len(data["characters"]) > 1:
        for i, left in enumerate(data["characters"]):
            for right in data["characters"][i + 1 :]:
                key = pair_key(left["id"], right["id"])
                rel = data["relationships"].setdefault(
                    key,
                    {
                        "trust_level": 50.0,
                        "sexual_tension": 30.0,
                        "respect_level": 50.0,
                        "dominance_dynamic": 0.0,
                    },
                )
                rel["trust_level"] = float(clamp(float(rel.get("trust_level", 50.0)), 0, 100))
                rel["sexual_tension"] = float(clamp(float(rel.get("sexual_tension", 30.0)), 0, 100))
                rel["respect_level"] = float(clamp(float(rel.get("respect_level", 50.0)), 0, 100))
                rel["dominance_dynamic"] = float(clamp(float(rel.get("dominance_dynamic", 0.0)), -50, 50))

    data["current_stage"] = int(clamp(float(data.get("current_stage", scenario.get("start_stage", 1))), 1, 6))
    data["freeze_stage"] = bool(data.get("freeze_stage", False))
    manual = data.get("manual_stage_override")
    data["manual_stage_override"] = int(clamp(float(manual), 1, 6)) if manual is not None else None

    data["tension"] = float(clamp(float(data.get("tension", scenario["initial_tension"])), 0, 100))
    data["risk"] = float(clamp(float(data.get("risk", scenario["initial_risk"])), 0, 100))

    data.setdefault("history", [])
    data.setdefault("validator_logs", [])
    data["turn_index"] = int(data.get("turn_index", 0))
    data["scene_started"] = bool(data.get("scene_started", False))

    data.setdefault("config", {})
    data["config"].setdefault("rolling_memory_size", 8)
    data["config"].setdefault("max_rewrites", 2)
    data["config"].setdefault("npc_responses_per_turn", 2)
    data["config"].setdefault("npc_auto_rounds_on_open", 2)
    data["config"].setdefault("npc_auto_rounds_after_user", 2)
    data["config"].setdefault("npc_auto_rounds_on_advance", 2)
    data["config"]["npc_responses_per_turn"] = int(clamp(float(data["config"].get("npc_responses_per_turn", 2)), 1, 10))
    data["config"]["npc_auto_rounds_on_open"] = int(clamp(float(data["config"].get("npc_auto_rounds_on_open", 2)), 1, 8))
    data["config"]["npc_auto_rounds_after_user"] = int(clamp(float(data["config"].get("npc_auto_rounds_after_user", 2)), 1, 8))
    data["config"]["npc_auto_rounds_on_advance"] = int(clamp(float(data["config"].get("npc_auto_rounds_on_advance", 2)), 1, 8))

    return data


class StateManager:
    def __init__(self, state_path: str | Path | None = None) -> None:
        self.state_path = Path(state_path) if state_path else None

    def load(self, path: str | Path | None = None) -> dict[str, Any]:
        target = Path(path) if path else self.state_path
        if not target or not target.exists():
            return create_default_state()
        data = json.loads(target.read_text(encoding="utf-8"))
        return normalize_state(data)

    def save(self, state: dict[str, Any], path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.state_path
        if target is None:
            raise ValueError("No state path provided")
        normalized = normalize_state(state)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(normalized, indent=2, ensure_ascii=True), encoding="utf-8")
        return target

    def dump_json(self, state: dict[str, Any]) -> str:
        return json.dumps(normalize_state(state), indent=2, ensure_ascii=True)

    def load_json_text(self, text: str) -> dict[str, Any]:
        return normalize_state(json.loads(text))


def rebuild_relationships_for_characters(state: dict[str, Any]) -> dict[str, Any]:
    state = normalize_state(state)
    current = state.get("relationships", {})
    rebuilt = build_relationships(state["characters"])
    for key, base in rebuilt.items():
        if key in current:
            for rel_key in base:
                base[rel_key] = current[key].get(rel_key, base[rel_key])
        rebuilt[key] = base
    state["relationships"] = rebuilt
    return state


def stage_label(stage: int) -> str:
    return STAGE_NAMES.get(stage, f"Stage {stage}")
