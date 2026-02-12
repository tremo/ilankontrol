from __future__ import annotations

from itertools import combinations
from typing import Dict, List

STAGE_NAMES: Dict[int, str] = {
    1: "Neutral",
    2: "Social Tension",
    3: "Suggestive Layer",
    4: "Private Energy",
    5: "Escalation Window",
    6: "Peak",
}

TRAIT_NAMES: List[str] = [
    "dominance",
    "libido",
    "risk_tolerance",
    "deniability",
    "impulsivity",
    "jealousy_trigger",
    "escalation_speed",
    "emotional_inertia",
]


def default_trait(base: int = 50) -> dict:
    return {
        "base": base,
        "current": float(base),
        "volatility": 0.35,
        "dynamic_enabled": True,
    }


def make_character(char_id: str, name: str, role: str, description: str, base: int = 50) -> dict:
    return {
        "id": char_id,
        "name": name,
        "role": role,
        "description": description,
        "traits": {trait: default_trait(base) for trait in TRAIT_NAMES},
    }


def pair_key(left_id: str, right_id: str) -> str:
    ordered = sorted([left_id, right_id])
    return f"{ordered[0]}|{ordered[1]}"


def build_relationships(characters: list[dict]) -> dict:
    relationships: dict = {}
    for a, b in combinations(characters, 2):
        relationships[pair_key(a["id"], b["id"])] = {
            "trust_level": 50.0,
            "sexual_tension": 30.0,
            "respect_level": 50.0,
            "dominance_dynamic": 0.0,
        }
    return relationships


def create_default_state() -> dict:
    characters = [
        make_character("user", "You", "user_controlled", "Player controlled character", 50),
        make_character("npc_1", "Ari", "npc", "Adaptive counterpart in simulation", 52),
    ]

    return {
        "scenario": {
            "name": "Default Scenario",
            "start_stage": 1,
            "environment_description": "A social indoor setting with ambient noise.",
            "style_notes": "",
            "style_examples": [],
            "safe_word": "",
            "flags": {
                "substance_effect": False,
            },
            "initial_tension": 25.0,
            "initial_risk": 15.0,
        },
        "characters": characters,
        "relationships": build_relationships(characters),
        "current_stage": 1,
        "freeze_stage": False,
        "manual_stage_override": None,
        "tension": 25.0,
        "risk": 15.0,
        "history": [],
        "validator_logs": [],
        "turn_index": 0,
        "scene_started": False,
        "config": {
            "rolling_memory_size": 8,
            "max_rewrites": 2,
            "npc_responses_per_turn": 2,
            "npc_auto_rounds_on_open": 2,
            "npc_auto_rounds_after_user": 2,
            "npc_auto_rounds_on_advance": 2,
        },
    }
