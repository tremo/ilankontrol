from __future__ import annotations

from typing import Any

from app.default_state import STAGE_NAMES


def summarize_traits(character: dict[str, Any]) -> str:
    parts: list[str] = []
    for trait_name, values in character.get("traits", {}).items():
        current = round(float(values.get("current", 50.0)), 1)
        base = int(values.get("base", 50))
        parts.append(f"{trait_name}: current={current}, base={base}")
    joined = "; ".join(parts)
    return f"{character['name']} ({character['role']}): {joined}"


def build_system_prompt(state: dict[str, Any], active_npc: dict[str, Any]) -> str:
    stage = int(state["current_stage"])
    stage_name = STAGE_NAMES.get(stage, f"Stage {stage}")
    scenario = state["scenario"]

    relationships = []
    for key, rel in state.get("relationships", {}).items():
        relationships.append(
            f"{key} -> trust={rel['trust_level']:.1f}, tension={rel['sexual_tension']:.1f}, respect={rel['respect_level']:.1f}, dom_dynamic={rel['dominance_dynamic']:.1f}"
        )

    character_blocks = [summarize_traits(c) for c in state.get("characters", [])]

    constraints = [
        "This is a stateful simulation engine output, not a generic chatbot reply.",
        "Keep behavior indirect and natural. Do not explain hidden motivations explicitly.",
        "Avoid meta self-analysis and avoid saying why you are acting in detail.",
        "Maintain deniability at early stages.",
        "Avoid abrupt escalation and any explicit content before allowed stages.",
        "Keep response concise and grounded in immediate scene context.",
    ]

    stage_rules = [
        "Stage 1-2: no direct declaration of intent.",
        "Stage 1-3: avoid explicit language.",
        "Escalation must be gradual and state-driven.",
    ]

    style_notes = str(scenario.get("style_notes", "")).strip()
    style_examples = scenario.get("style_examples", [])[:10]
    style_block = ""
    if style_notes:
        style_block += f"\nStyle notes:\n{style_notes}\n"
    if style_examples:
        style_block += "Style examples (tone reference only, not current events):\n- " + "\n- ".join(style_examples) + "\n"

    character_block = "\n- ".join(character_blocks) if character_blocks else "None"
    relationship_block = "\n- ".join(relationships) if relationships else "None"
    behavior_block = "\n- ".join(constraints + stage_rules)

    return (
        f"You are simulating character: {active_npc['name']}\n"
        f"Scenario: {scenario['name']}\n"
        f"Environment: {scenario['environment_description']}\n"
        f"Current stage: {stage} ({stage_name})\n"
        f"Current tension: {state['tension']:.1f}/100\n"
        f"Current risk: {state['risk']:.1f}/100\n\n"
        f"Character state:\n- {character_block}\n\n"
        f"Relationships:\n- {relationship_block}\n\n"
        f"{style_block}"
        f"Behavior constraints:\n- {behavior_block}"
    )


def rolling_history(state: dict[str, Any]) -> list[dict[str, str]]:
    max_items = int(state.get("config", {}).get("rolling_memory_size", 8))
    history = state.get("history", [])[-max_items:]
    msgs: list[dict[str, str]] = []
    for item in history:
        speaker = item.get("speaker", "unknown")
        text = item.get("message", "")
        msgs.append({"role": "user" if speaker == "user" else "assistant", "content": f"{speaker}: {text}"})
    return msgs


def build_planner_prompt(state: dict[str, Any], user_message: str, active_npc: dict[str, Any]) -> str:
    return (
        f"Generate a one-line tactical response plan for {active_npc['name']} in this turn. "
        f"Current stage={state['current_stage']}, tension={state['tension']:.1f}, risk={state['risk']:.1f}. "
        f"User action: {user_message} "
        "The plan must preserve realism, deniability, and gradual progression."
    )


def build_generator_prompt(state: dict[str, Any], user_message: str, plan: str) -> str:
    return (
        f"Plan: {plan}\n"
        f"User action: {user_message}\n"
        "Respond as the active NPC in 1-3 sentences, subtle and context-aware."
    )
