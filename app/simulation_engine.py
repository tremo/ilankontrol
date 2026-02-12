from __future__ import annotations

import random
import re
from typing import Any

from app.default_state import STAGE_NAMES, pair_key
from app.llm_client import LLMClient
from app.prompting import build_generator_prompt, build_planner_prompt, build_system_prompt
from app.state_manager import clamp, normalize_state
from app.validator import ValidationIssue, apply_safe_deltas, rewrite_with_rules, validate_output


class SimulationEngine:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def start_scene(self, state: dict[str, Any], user_actor_id: str) -> tuple[dict[str, Any], list[dict[str, str]], list[ValidationIssue]]:
        state = normalize_state(state)
        if state.get("scene_started") and state.get("history"):
            return state, [], []

        actor = self._find_character(state, user_actor_id)
        if actor is None:
            actor = next((c for c in state["characters"] if c["role"] == "user_controlled"), state["characters"][0])

        npcs = self._npc_order_for_actor(state, actor["id"])
        if not npcs:
            return state, [], []

        turn_id = int(state.get("turn_index", 0)) + 1
        outputs: list[dict[str, str]] = []
        all_issues: list[ValidationIssue] = []

        scene_line = self._build_scene_opening(state)
        outputs.append({"speaker_id": "scene", "speaker": "Scene", "text": scene_line})
        state["history"].append(
            {
                "turn": turn_id,
                "speaker": "scene",
                "message": scene_line,
                "meta": {"role": "narrator", "stage": state["current_stage"], "opening": True},
            }
        )

        opening_tension_delta = 1.0 + (1.0 if state["scenario"].get("flags", {}).get("substance_effect") else 0.0)
        opening_risk_delta = 0.6 + (1.1 if state["scenario"].get("flags", {}).get("substance_effect") else 0.0)
        opening_tension_delta, opening_risk_delta = apply_safe_deltas(state, opening_tension_delta, opening_risk_delta)
        state["tension"] = clamp(state["tension"] + opening_tension_delta, 0, 100)
        state["risk"] = clamp(state["risk"] + opening_risk_delta, 0, 100)
        self._apply_trait_drift(state, turn_id, "scene opening")
        self._update_stage(state)

        rounds = int(state.get("config", {}).get("npc_auto_rounds_on_open", 2))
        round_outputs, round_issues = self._run_npc_rounds(
            state=state,
            actor=actor,
            npcs=npcs,
            turn_id=turn_id,
            rounds=rounds,
            base_user_message="Scene just opened. Continue naturally and establish the immediate social dynamics.",
            opening=True,
        )
        outputs.extend(round_outputs)
        all_issues.extend(round_issues)

        state["turn_index"] = turn_id
        state["scene_started"] = True
        return state, outputs, all_issues

    def run_turn(
        self, state: dict[str, Any], user_actor_id: str, user_message: str
    ) -> tuple[dict[str, Any], list[dict[str, str]], list[ValidationIssue]]:
        state = normalize_state(state)
        user_message = (user_message or "").strip()
        if not user_message:
            return state, [], []

        actor = self._find_character(state, user_actor_id)
        if actor is None:
            actor = next((c for c in state["characters"] if c["role"] == "user_controlled"), state["characters"][0])

        npcs = self._npc_order_for_actor(state, actor["id"])
        if not npcs:
            return state, [], []

        primary_npc = npcs[0]
        turn_id = int(state.get("turn_index", 0)) + 1

        state["history"].append(
            {
                "turn": turn_id,
                "speaker": actor["id"],
                "message": user_message,
                "meta": {"role": actor["role"]},
            }
        )

        tension_delta, risk_delta = self._compute_deltas(state, actor, primary_npc, user_message)
        tension_delta, risk_delta = apply_safe_deltas(state, tension_delta, risk_delta)
        state["tension"] = clamp(state["tension"] + tension_delta, 0, 100)
        state["risk"] = clamp(state["risk"] + risk_delta, 0, 100)

        self._apply_trait_drift(state, turn_id, user_message)
        for idx, npc in enumerate(npcs):
            scale = 1.0 if idx == 0 else max(0.25, 0.7 - idx * 0.2)
            self._apply_relationship_drift(state, actor, npc, tension_delta * scale, risk_delta * scale)
        self._update_stage(state)

        rounds = int(state.get("config", {}).get("npc_auto_rounds_after_user", 2))
        outputs, all_issues = self._run_npc_rounds(
            state=state,
            actor=actor,
            npcs=npcs,
            turn_id=turn_id,
            rounds=rounds,
            base_user_message=user_message,
            opening=False,
        )

        state["turn_index"] = turn_id
        state["scene_started"] = True
        return state, outputs, all_issues

    def advance_scene(
        self, state: dict[str, Any], user_actor_id: str, guidance: str = ""
    ) -> tuple[dict[str, Any], list[dict[str, str]], list[ValidationIssue]]:
        state = normalize_state(state)

        actor = self._find_character(state, user_actor_id)
        if actor is None:
            actor = next((c for c in state["characters"] if c["role"] == "user_controlled"), state["characters"][0])

        npcs = self._npc_order_for_actor(state, actor["id"])
        if not npcs:
            return state, [], []

        turn_id = int(state.get("turn_index", 0)) + 1
        base_message = guidance.strip() if guidance.strip() else "Advance the scene naturally without explicit jumps."

        rounds = int(state.get("config", {}).get("npc_auto_rounds_on_advance", 2))
        outputs, all_issues = self._run_npc_rounds(
            state=state,
            actor=actor,
            npcs=npcs,
            turn_id=turn_id,
            rounds=rounds,
            base_user_message=base_message,
            opening=False,
        )

        state["turn_index"] = turn_id
        state["scene_started"] = True
        return state, outputs, all_issues

    def _run_npc_rounds(
        self,
        state: dict[str, Any],
        actor: dict[str, Any],
        npcs: list[dict[str, Any]],
        turn_id: int,
        rounds: int,
        base_user_message: str,
        opening: bool,
    ) -> tuple[list[dict[str, str]], list[ValidationIssue]]:
        outputs: list[dict[str, str]] = []
        all_issues: list[ValidationIssue] = []
        max_npcs = self._npc_response_count(state, npcs)
        max_rounds = max(1, rounds)

        recent_prompt = base_user_message
        stage_origin = int(state.get("current_stage", 1))
        tension_origin = float(state.get("tension", 0.0))

        for round_idx in range(max_rounds):
            round_outputs: list[dict[str, str]] = []
            for npc in npcs[:max_npcs]:
                stage_before = int(state["current_stage"])
                tension_before = float(state["tension"])

                response, issues, rewrite_count, plan = self._generate_npc_turn(
                    state=state,
                    npc=npc,
                    user_message=recent_prompt,
                    stage_before=stage_before,
                    tension_before=tension_before,
                    opening=opening and round_idx == 0,
                )

                state["history"].append(
                    {
                        "turn": turn_id,
                        "speaker": npc["id"],
                        "message": response,
                        "meta": {
                            "role": npc["role"],
                            "plan": plan,
                            "stage": state["current_stage"],
                            "validator_rewrites": rewrite_count,
                            "round": round_idx + 1,
                            "opening": bool(opening and round_idx == 0),
                        },
                    }
                )

                self._append_validator_log(state, turn_id, npc["id"], issues, rewrite_count)
                self._apply_npc_output_effects(state, actor, npc, response, turn_id, round_idx)

                out = {"speaker_id": npc["id"], "speaker": npc["name"], "text": response}
                outputs.append(out)
                round_outputs.append(out)
                all_issues.extend(issues)

            if round_outputs:
                recent_prompt = " ".join(item["text"] for item in round_outputs)

            if self._is_meaningful_stop(state, stage_origin, tension_origin, round_idx):
                break

        return outputs, all_issues

    def _is_meaningful_stop(self, state: dict[str, Any], stage_origin: int, tension_origin: float, round_idx: int) -> bool:
        if round_idx < 1:
            return False

        stage_shift = int(state.get("current_stage", 1)) - stage_origin
        tension_shift = abs(float(state.get("tension", 0.0)) - tension_origin)
        return stage_shift >= 1 or tension_shift >= 6.0

    def _apply_npc_output_effects(
        self,
        state: dict[str, Any],
        actor: dict[str, Any],
        npc: dict[str, Any],
        response_text: str,
        turn_id: int,
        round_idx: int,
    ) -> None:
        tone = self._message_tone(response_text)
        escalation_drive = (
            float(npc["traits"]["escalation_speed"]["current"])
            + float(npc["traits"]["impulsivity"]["current"])
            + float(npc["traits"]["libido"]["current"])
        ) / 3.0
        deniability = float(npc["traits"]["deniability"]["current"])

        tension_delta = 0.45 + max(0.0, tone) * 0.85 + (escalation_drive - 50.0) / 60.0
        risk_delta = 0.20 + max(0.0, tone) * 0.55 + (escalation_drive - deniability) / 90.0

        round_decay = max(0.45, 1.0 - round_idx * 0.15)
        tension_delta *= round_decay
        risk_delta *= round_decay

        tension_delta, risk_delta = apply_safe_deltas(state, tension_delta, risk_delta)
        state["tension"] = clamp(state["tension"] + tension_delta, 0, 100)
        state["risk"] = clamp(state["risk"] + risk_delta, 0, 100)

        self._apply_relationship_drift(state, actor, npc, tension_delta * 0.9, risk_delta * 0.9)
        self._apply_trait_drift(state, turn_id, response_text)
        self._update_stage(state)

    def _generate_npc_turn(
        self,
        state: dict[str, Any],
        npc: dict[str, Any],
        user_message: str,
        stage_before: int,
        tension_before: float,
        opening: bool,
    ) -> tuple[str, list[ValidationIssue], int, str]:
        system_prompt = build_system_prompt(state, npc)

        if opening:
            plan_prompt = (
                f"Create one-line opening tactic for {npc['name']} at stage={state['current_stage']}. "
                "No explicit escalation, no long exposition, one subtle social move."
            )
            plan = self.llm.planner(system_prompt, plan_prompt)
            gen_prompt = (
                f"Plan: {plan}\n"
                "Generate first in-scene move in 1-2 sentences. "
                "Blend a small action and a short natural line."
            )
        else:
            plan = self.llm.planner(system_prompt, build_planner_prompt(state, user_message, npc))
            gen_prompt = build_generator_prompt(state, user_message, plan)

        response = self.llm.generator(system_prompt, gen_prompt)

        stage_after = int(state["current_stage"])
        issues = validate_output(
            state=state,
            text=response,
            stage_before_turn=stage_before,
            stage_after_turn=stage_after,
            tension_delta=state["tension"] - tension_before,
        )
        issues.extend(self._llm_validator_issues(system_prompt, response, stage_before, stage_after))

        rewritten = response
        rewrite_count = 0
        max_rewrites = int(state.get("config", {}).get("max_rewrites", 2))
        while issues and rewrite_count < max_rewrites:
            rewritten = rewrite_with_rules(rewritten, state["current_stage"])
            rewrite_count += 1
            issues = validate_output(
                state=state,
                text=rewritten,
                stage_before_turn=stage_before,
                stage_after_turn=stage_after,
                tension_delta=state["tension"] - tension_before,
            )
            issues.extend(self._llm_validator_issues(system_prompt, rewritten, stage_before, stage_after))

        return rewritten, issues, rewrite_count, plan

    def _append_validator_log(
        self,
        state: dict[str, Any],
        turn_id: int,
        speaker_id: str,
        issues: list[ValidationIssue],
        rewrite_count: int,
    ) -> None:
        logs = state.get("validator_logs", [])
        logs.append(
            {
                "turn": turn_id,
                "speaker": speaker_id,
                "stage": state["current_stage"],
                "stage_name": STAGE_NAMES.get(state["current_stage"], "Unknown"),
                "issues": [issue.__dict__ for issue in issues],
                "rewrite_count": rewrite_count,
            }
        )
        state["validator_logs"] = logs[-120:]

    def _find_character(self, state: dict[str, Any], char_id: str) -> dict[str, Any] | None:
        for char in state.get("characters", []):
            if char.get("id") == char_id:
                return char
        return None

    def _npc_order_for_actor(self, state: dict[str, Any], actor_id: str) -> list[dict[str, Any]]:
        npcs = [c for c in state["characters"] if c["id"] != actor_id]
        scored: list[tuple[float, dict[str, Any]]] = []
        for npc in npcs:
            key = pair_key(actor_id, npc["id"])
            rel = state.get("relationships", {}).get(key, {})
            tension = float(rel.get("sexual_tension", 0.0))
            trust = float(rel.get("trust_level", 0.0))
            drive = (
                float(npc["traits"]["libido"]["current"])
                + float(npc["traits"]["escalation_speed"]["current"])
                + float(npc["traits"]["impulsivity"]["current"])
            ) / 3.0
            score = tension * 0.7 + trust * 0.2 + drive * 0.1
            scored.append((score, npc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored]

    def _npc_response_count(self, state: dict[str, Any], npcs: list[dict[str, Any]]) -> int:
        desired = int(state.get("config", {}).get("npc_responses_per_turn", 2))
        return max(1, min(desired, len(npcs)))

    def _build_scene_opening(self, state: dict[str, Any]) -> str:
        scenario = state.get("scenario", {})
        env = str(scenario.get("environment_description", "")).strip()
        stage_name = STAGE_NAMES.get(int(state.get("current_stage", 1)), "Neutral")

        base = env if env else "The scene opens in a socially charged environment with subtle undercurrents."
        return (
            f"{base} Stage tone is {stage_name.lower()}, tension {state['tension']:.1f}/100 and risk {state['risk']:.1f}/100. "
            "Characters settle into position and the first small moves begin."
        )

    def _compute_deltas(
        self,
        state: dict[str, Any],
        actor: dict[str, Any],
        active_npc: dict[str, Any],
        user_message: str,
    ) -> tuple[float, float]:
        text = user_message.lower()

        punctuation_push = min(user_message.count("!") * 0.8, 3.0)
        directive_push = 2.0 if re.search(r"\b(now|come on|do it|closer)\b", text) else 0.0
        hedging_pull = -1.5 if re.search(r"\bmaybe|perhaps|not sure|later\b", text) else 0.0

        npc_traits = active_npc["traits"]
        escalation_drive = (
            float(npc_traits["escalation_speed"]["current"])
            + float(npc_traits["impulsivity"]["current"])
            + float(npc_traits["libido"]["current"])
        ) / 3.0
        containment_drive = (
            float(npc_traits["deniability"]["current"]) + float(npc_traits["emotional_inertia"]["current"])
        ) / 2.0

        tension_delta = 1.2 + punctuation_push + directive_push + hedging_pull
        tension_delta += (escalation_drive - 50.0) / 12.0
        tension_delta -= (containment_drive - 50.0) / 18.0

        if state["scenario"].get("flags", {}).get("substance_effect"):
            tension_delta += 1.2

        risk_base = (
            float(npc_traits["risk_tolerance"]["current"]) + float(npc_traits["impulsivity"]["current"])
        ) / 2.0
        risk_delta = 0.8 + punctuation_push * 0.4 + directive_push * 0.8
        risk_delta += (risk_base - 50.0) / 15.0
        risk_delta -= float(npc_traits["deniability"]["current"] - 50.0) / 20.0

        return tension_delta, risk_delta

    def _apply_trait_drift(self, state: dict[str, Any], turn_id: int, user_message: str) -> None:
        tone = self._message_tone(user_message)
        for char in state.get("characters", []):
            traits = char.get("traits", {})
            inertia = float(traits.get("emotional_inertia", {}).get("current", 50.0)) / 100.0
            for trait_name, trait in traits.items():
                if not trait.get("dynamic_enabled", True):
                    continue

                base = float(trait.get("base", 50.0))
                current = float(trait.get("current", base))
                volatility = float(trait.get("volatility", 0.35))

                random.seed(f"{turn_id}:{char['id']}:{trait_name}")
                noise = random.uniform(-1.0, 1.0)

                stage_pressure = float(state["tension"]) / 100.0
                drift_gain = stage_pressure * (0.4 + volatility) * (1.0 - inertia * 0.7)

                direction = self._trait_direction(trait_name, tone, state)
                reversion = (base - current) * 0.02

                delta = (direction + noise * 0.25) * drift_gain * 8.0 + reversion
                trait["current"] = clamp(current + delta, 0, 100)

    def _trait_direction(self, trait_name: str, tone: float, state: dict[str, Any]) -> float:
        tension = float(state["tension"]) / 100.0
        risk = float(state["risk"]) / 100.0

        if trait_name in {"dominance", "libido", "escalation_speed", "impulsivity"}:
            return 0.3 + tone * 0.6 + tension * 0.4
        if trait_name == "risk_tolerance":
            return 0.2 + tone * 0.2 + risk * 0.6
        if trait_name == "deniability":
            return 0.25 - tension * 0.5 - tone * 0.3
        if trait_name == "jealousy_trigger":
            return 0.1 + risk * 0.7 - tone * 0.2
        if trait_name == "emotional_inertia":
            return -0.2 + (0.5 - tension) * 0.4
        return 0.0

    def _apply_relationship_drift(
        self,
        state: dict[str, Any],
        actor: dict[str, Any],
        active_npc: dict[str, Any],
        tension_delta: float,
        risk_delta: float,
    ) -> None:
        key = pair_key(actor["id"], active_npc["id"])
        rel = state.get("relationships", {}).get(key)
        if rel is None:
            return

        actor_dom = float(actor["traits"]["dominance"]["current"])
        npc_dom = float(active_npc["traits"]["dominance"]["current"])

        rel["sexual_tension"] = clamp(float(rel["sexual_tension"]) + tension_delta * 0.8, 0, 100)
        rel["trust_level"] = clamp(float(rel["trust_level"]) + (2.0 - risk_delta) * 0.5, 0, 100)
        rel["respect_level"] = clamp(float(rel["respect_level"]) + (npc_dom - 50.0) * 0.04, 0, 100)
        rel["dominance_dynamic"] = clamp(float(rel["dominance_dynamic"]) + (npc_dom - actor_dom) * 0.05, -50, 50)

    def _update_stage(self, state: dict[str, Any]) -> None:
        manual = state.get("manual_stage_override")
        current = int(state.get("current_stage", 1))

        if manual is not None:
            state["current_stage"] = int(clamp(float(manual), 1, 6))
            return

        if state.get("freeze_stage"):
            return

        target = self._stage_from_tension(state["tension"])
        if target > current + 1:
            target = current + 1
        elif target < current - 1:
            target = current - 1
        state["current_stage"] = int(clamp(target, 1, 6))

    def _stage_from_tension(self, tension: float) -> int:
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

    def _message_tone(self, user_message: str) -> float:
        text = user_message.lower()
        positive = len(re.findall(r"\b(yes|closer|more|again|now|sure)\b", text))
        cautious = len(re.findall(r"\b(wait|slow|later|maybe|stop|pause)\b", text))
        return clamp((positive - cautious) / 4.0, -1.0, 1.0)

    def _llm_validator_issues(
        self, system_prompt: str, text: str, stage_before: int, stage_after: int
    ) -> list[ValidationIssue]:
        validator_prompt = (
            "Check this response for violations and answer ONLY 'OK' or 'VIOLATION: <short reason>'.\n"
            f"Stage before={stage_before}, stage after={stage_after}.\n"
            f"Response: {text}"
        )
        result = self.llm.validator(system_prompt, validator_prompt).strip()
        if result.upper().startswith("VIOLATION"):
            reason = result.split(":", 1)[1].strip() if ":" in result else "LLM validator flagged a violation."
            return [
                ValidationIssue(
                    code="llm_validator_flag",
                    message=reason,
                    severity="medium",
                )
            ]
        return []
