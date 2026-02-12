from __future__ import annotations

import copy
import re
import unicodedata
from typing import Any

from app.default_state import TRAIT_NAMES, build_relationships, create_default_state
from app.state_manager import clamp, normalize_state

START_MARKERS = [
    "Senaryo surdan basliyor:",
    "Senaryo su sekilde basliyor:",
    "Scenario starts:",
    "Story starts here:",
]


def _slugify(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    ascii_value = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value).strip("_").lower()
    return ascii_value or "char"


def _ascii_fold(value: str) -> str:
    tr_map = str.maketrans(
        {
            "ı": "i",
            "İ": "I",
            "ş": "s",
            "Ş": "S",
            "ğ": "g",
            "Ğ": "G",
            "ü": "u",
            "Ü": "U",
            "ö": "o",
            "Ö": "O",
            "ç": "c",
            "Ç": "C",
        }
    )
    value = value.translate(tr_map)
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


def _fold_with_index_map(value: str) -> tuple[str, list[int]]:
    folded_parts: list[str] = []
    index_map: list[int] = []
    for idx, ch in enumerate(value):
        folded = _ascii_fold(ch)
        if not folded:
            continue
        folded_parts.append(folded)
        for _ in folded:
            index_map.append(idx)
    return "".join(folded_parts), index_map


def _default_trait_block(base: int) -> dict[str, Any]:
    return {
        "base": int(clamp(base, 0, 100)),
        "current": float(clamp(base, 0, 100)),
        "volatility": 0.35,
        "dynamic_enabled": True,
    }


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _infer_traits(description: str) -> dict[str, dict[str, Any]]:
    lower = description.lower()
    scores: dict[str, float] = {name: 50.0 for name in TRAIT_NAMES}

    if _contains_any(lower, ["dominant", "agresif", "kontrolcu", "alfa", "hoyrat"]):
        scores["dominance"] += 24
        scores["escalation_speed"] += 8

    if _contains_any(lower, ["libido", "azmis", "azgin", "high libido", "cok yuksek"]):
        scores["libido"] += 30
        scores["impulsivity"] += 10

    if _contains_any(lower, ["kiskanc degil", "kiskanc degildir", "kiskan", "kiskan degil"]):
        scores["jealousy_trigger"] -= 20

    if _contains_any(lower, ["temkinli", "careful"]):
        scores["risk_tolerance"] -= 12
        scores["deniability"] += 8

    if _contains_any(lower, ["inkar", "plausible deniability", "dolayli", "saka kisvesi"]):
        scores["deniability"] += 26

    if _contains_any(lower, ["hayir demekte zor", "itirazlarini dinlemez", "boyun eger"]):
        scores["dominance"] -= 14
        scores["emotional_inertia"] += 12

    if _contains_any(lower, ["dayanamayabilir", "hizli", "ani", "gerilim"]):
        scores["escalation_speed"] += 14
        scores["impulsivity"] += 12

    if _contains_any(lower, ["cool", "sakin", "olgun"]):
        scores["emotional_inertia"] += 10
        scores["impulsivity"] -= 8

    volatility = 0.45 if _contains_any(lower, ["cok", "asiri", "hoyrat", "agresif"]) else 0.35

    traits: dict[str, dict[str, Any]] = {}
    for name in TRAIT_NAMES:
        base = int(clamp(round(scores[name]), 0, 100))
        traits[name] = {
            "base": base,
            "current": float(base),
            "volatility": volatility,
            "dynamic_enabled": True,
        }
    return traits


def _parse_character_blocks(text: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"^[ \t]*[\*\-][ \t]*([^:\n]+):[ \t]*(.*?)(?=^[ \t]*[\*\-][ \t]*[^:\n]+:[ \t]*|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    characters: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for idx, match in enumerate(matches):
        raw_name = re.sub(r"\s+", " ", match.group(1)).strip()
        description = re.sub(r"\s+", " ", match.group(2)).strip()
        if not raw_name:
            continue

        display_name = re.sub(r"\([^)]*\)", "", raw_name).strip() or raw_name
        role = "user_controlled" if "ben" in raw_name.lower() else "npc"

        char_id = _slugify(display_name)
        if char_id in used_ids:
            char_id = f"{char_id}_{idx + 1}"
        used_ids.add(char_id)

        character = {
            "id": char_id,
            "name": display_name,
            "role": role,
            "description": description,
            "traits": _infer_traits(description),
        }
        characters.append(character)

    if characters and not any(char["role"] == "user_controlled" for char in characters):
        characters[0]["role"] = "user_controlled"

    return characters


def _extract_safe_word(text: str) -> str:
    quoted_patterns = [
        re.compile(r"(?:safe\s*word|guvenli\s*kelime\w*)[^\n]{0,80}[\"']([^\"']{1,64})[\"']", re.IGNORECASE),
    ]
    for pattern in quoted_patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()

    raw_patterns = [
        re.compile(r"safe\s*word\s*[:=]?\s*([A-Za-z0-9_-]{2,64})", re.IGNORECASE),
        re.compile(r"guvenli\s*kelime\w*\s*[:=]?\s*([A-Za-z0-9_-]{2,64})", re.IGNORECASE),
    ]
    for pattern in raw_patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return ""


def _extract_environment(text: str) -> str:
    marker_index, marker = _find_start_marker(text)
    if marker_index is not None and marker:
        snippet = text[marker_index + len(marker) :].strip()
        return re.sub(r"\s+", " ", snippet)[:3000]

    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:1800]


def _find_start_marker(text: str) -> tuple[int | None, str | None]:
    folded, idx_map = _fold_with_index_map(text)
    folded = folded.lower()
    hits: list[tuple[int, str]] = []
    for marker in START_MARKERS:
        idx = folded.find(marker.lower())
        if idx >= 0:
            if idx < len(idx_map):
                hits.append((idx_map[idx], marker))
    if not hits:
        return None, None
    hits.sort(key=lambda item: item[0])
    return hits[0]


def _extract_chat_lines(text: str, characters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    id_by_name = {_slugify(char["name"]): char["id"] for char in characters}
    chat_pattern = re.compile(r"^\[(\d{1,2}:\d{2}),\s*(\d{2}/\d{2}/\d{4})\]\s*([^:]+):\s*(.*)$", re.MULTILINE)

    history: list[dict[str, Any]] = []
    for match in chat_pattern.finditer(text):
        speaker_name = match.group(3).strip()
        message = match.group(4).strip()
        slug = _slugify(re.sub(r"\([^)]*\)", "", speaker_name))
        speaker_id = id_by_name.get(slug, slug)

        if not message:
            continue

        history.append(
            {
                "speaker": speaker_id,
                "message": message,
                "meta": {
                    "source": "imported_chat",
                    "time": match.group(1),
                    "date": match.group(2),
                    "speaker_label": speaker_name,
                },
            }
        )

    return history[-24:]


def _as_turn_history(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        item = dict(line)
        item["turn"] = idx
        history.append(item)
    return history


def _style_examples_from_lines(lines: list[dict[str, Any]]) -> list[str]:
    examples: list[str] = []
    for line in lines[-20:]:
        label = str(line.get("meta", {}).get("speaker_label", line.get("speaker", "unknown"))).strip()
        message = str(line.get("message", "")).strip()
        if message:
            examples.append(f"{label}: {message}")
    return examples[-16:]


def _infer_tension_and_risk(text: str) -> tuple[float, float, bool]:
    lower = text.lower()
    tension = 25.0
    risk = 15.0
    substance_effect = False

    if _contains_any(lower, ["gerilim", "kiskan", "flort", "atmosfer", "seksuel", "yaklasim"]):
        tension += 14

    if _contains_any(lower, ["bira", "alkol", "ot", "kokain", "uyusturucu", "madde"]):
        tension += 9
        risk += 22
        substance_effect = True

    if _contains_any(lower, ["kavga", "gergin", "kriz", "risk"]):
        risk += 10

    return clamp(tension, 0, 100), clamp(risk, 0, 100), substance_effect


def _stage_from_tension(tension: float) -> int:
    if tension < 25:
        return 1
    if tension < 40:
        return 2
    if tension < 55:
        return 3
    if tension < 70:
        return 4
    if tension < 85:
        return 5
    return 6


def _apply_relationship_heuristics(state: dict[str, Any], raw_text: str) -> None:
    lower = raw_text.lower()
    char_by_id = {char["id"]: char for char in state["characters"]}

    for key, rel in state["relationships"].items():
        left_id, right_id = key.split("|")
        left = char_by_id[left_id]
        right = char_by_id[right_id]

        pair_text = f"{left['description']} {right['description']}".lower()

        trust = float(rel["trust_level"])
        tension = float(rel["sexual_tension"])
        respect = float(rel["respect_level"])

        if _contains_any(pair_text, ["en yakin", "aralarindan su sizmaz", "close friend"]):
            trust += 20
            respect += 8

        if "evli" in pair_text:
            trust += 14
            tension += 8

        if _contains_any(pair_text, ["flort", "gerilim", "libido", "azmis", "dominant"]):
            tension += 14

        if _contains_any(lower, ["kiskan", "kriz", "itiraz"]):
            trust -= 6

        left_dom = float(left["traits"]["dominance"]["base"])
        right_dom = float(right["traits"]["dominance"]["base"])
        rel["dominance_dynamic"] = clamp((right_dom - left_dom) * 0.35, -50, 50)
        rel["trust_level"] = clamp(trust, 0, 100)
        rel["sexual_tension"] = clamp(tension, 0, 100)
        rel["respect_level"] = clamp(respect, 0, 100)


def import_scenario_text(
    raw_text: str, base_state: dict[str, Any] | None = None, chat_to_history: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    text = (raw_text or "").strip()
    seed = normalize_state(copy.deepcopy(base_state) if base_state else create_default_state())
    if not text:
        return seed, {"characters": 0, "history": 0, "safe_word": "", "note": "No text provided."}

    characters = _parse_character_blocks(text)
    if characters:
        seed["characters"] = characters

    seed["relationships"] = build_relationships(seed["characters"])

    scenario = seed["scenario"]
    scenario["name"] = "Imported Scenario"
    scenario["environment_description"] = _extract_environment(text)
    scenario["safe_word"] = _extract_safe_word(text)

    tension, risk, substance = _infer_tension_and_risk(text)
    scenario["flags"]["substance_effect"] = substance
    scenario["initial_tension"] = float(tension)
    scenario["initial_risk"] = float(risk)

    seed["tension"] = float(tension)
    seed["risk"] = float(risk)
    seed["current_stage"] = _stage_from_tension(tension)
    seed["manual_stage_override"] = None
    seed["freeze_stage"] = False

    _apply_relationship_heuristics(seed, text)

    marker_index, marker = _find_start_marker(text)
    preface_text = text
    runtime_text = ""
    if marker_index is not None and marker:
        preface_text = text[:marker_index]
        runtime_text = text[marker_index + len(marker) :]

    style_lines = _extract_chat_lines(preface_text, seed["characters"])
    history_lines = _extract_chat_lines(runtime_text, seed["characters"])

    scenario["style_examples"] = _style_examples_from_lines(style_lines)
    scenario["style_notes"] = (
        "Keep message rhythm natural and concise; prefer indirect social behavior over explicit explanations."
    )

    if chat_to_history:
        history = _as_turn_history(_extract_chat_lines(text, seed["characters"]))
    else:
        history = _as_turn_history(history_lines)
    seed["history"] = history
    seed["turn_index"] = int(history[-1]["turn"]) if history else 0
    seed["scene_started"] = False

    seed["validator_logs"] = []

    summary = {
        "characters": len(seed["characters"]),
        "history": len(seed.get("history", [])),
        "style_examples": len(scenario.get("style_examples", [])),
        "safe_word": scenario.get("safe_word", ""),
        "stage": seed["current_stage"],
        "tension": seed["tension"],
        "risk": seed["risk"],
    }

    return normalize_state(seed), summary
