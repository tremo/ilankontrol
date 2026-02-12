from __future__ import annotations

import hashlib
import uuid

import streamlit as st

from app.default_state import STAGE_NAMES, TRAIT_NAMES, create_default_state
from app.llm_client import LLMClient, LLMConfig
from app.model_catalog import default_model_presets, fetch_models
from app.scenario_importer import import_scenario_text
from app.simulation_engine import SimulationEngine
from app.settings_store import LLM_SETTING_KEYS, clear_ui_settings, extract_llm_settings, load_ui_settings, save_ui_settings
from app.state_manager import StateManager, normalize_state, rebuild_relationships_for_characters, stage_label

st.set_page_config(page_title="Stateful Character Simulation MVP", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --bg1: #f6efe5;
      --bg2: #d8e6de;
      --panel: rgba(255,255,255,0.75);
      --text: #1e2a27;
      --accent: #0b7a75;
      --accent2: #cb5d32;
    }
    html, body, [data-testid="stAppViewContainer"] {
      background: radial-gradient(1200px 600px at 0% 0%, var(--bg2), transparent), linear-gradient(135deg, var(--bg1), #eef2ed);
      color: var(--text);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
    }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #f3ece2, #e8f1eb);
    }
    .sim-card {
      background: var(--panel);
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 14px;
      padding: 12px 14px;
      margin-bottom: 10px;
    }
    .sim-title {
      font-weight: 700;
      letter-spacing: 0.2px;
      color: var(--accent);
      margin-bottom: 4px;
    }
    .sim-sub {
      color: #3b4f4a;
      font-size: 0.95rem;
    }
    .warning-box {
      border-left: 4px solid var(--accent2);
      background: rgba(203,93,50,0.08);
      padding: 8px 10px;
      border-radius: 8px;
      margin: 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

manager = StateManager()

if "sim_state" not in st.session_state:
    st.session_state.sim_state = create_default_state()

if "last_outputs" not in st.session_state:
    st.session_state.last_outputs = []

if "last_issues" not in st.session_state:
    st.session_state.last_issues = []

if "import_summary" not in st.session_state:
    st.session_state.import_summary = None

if "model_catalog_cache" not in st.session_state:
    st.session_state.model_catalog_cache = {"fingerprint": "", "models": [], "error": ""}

if "ui_settings_loaded" not in st.session_state:
    saved_settings = load_ui_settings()
    for key, value in saved_settings.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state.ui_settings_loaded = True


def _scenario_widget_seed_payload(state: dict) -> dict:
    scenario = state.get("scenario", {})
    return {
        "scenario_name": str(scenario.get("name", "")),
        "scenario_env": str(scenario.get("environment_description", "")),
        "scenario_style_notes": str(scenario.get("style_notes", "")),
        "scenario_safeword": str(scenario.get("safe_word", "")),
        "scenario_style_examples": "\n".join(scenario.get("style_examples", [])),
        "scenario_start_stage": int(state.get("current_stage", scenario.get("start_stage", 1))),
        "scenario_initial_tension": float(scenario.get("initial_tension", 25.0)),
        "scenario_initial_risk": float(scenario.get("initial_risk", 15.0)),
        "scenario_substance": bool(scenario.get("flags", {}).get("substance_effect", False)),
    }


def queue_scenario_widget_state_seed(state: dict) -> None:
    st.session_state["_pending_scenario_widget_seed"] = _scenario_widget_seed_payload(state)


def apply_queued_scenario_widget_state_seed() -> None:
    payload = st.session_state.pop("_pending_scenario_widget_seed", None)
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        st.session_state[key] = value


def get_llm_config() -> LLMConfig:
    def _hash_secret(secret: str) -> str:
        if not secret:
            return ""
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:10]

    with st.sidebar:
        st.markdown("### LLM Settings")
        provider_options = ["rule_based", "openai", "anthropic", "gemini", "openai_compatible"]
        if st.session_state.get("llm_provider") not in provider_options:
            st.session_state["llm_provider"] = "rule_based"
        if not isinstance(st.session_state.get("llm_temperature", 0.6), (int, float)):
            st.session_state["llm_temperature"] = 0.6
        if not isinstance(st.session_state.get("llm_auto_fetch_models", True), bool):
            st.session_state["llm_auto_fetch_models"] = True
        if not isinstance(st.session_state.get("llm_verify_ssl", True), bool):
            st.session_state["llm_verify_ssl"] = True
        if not isinstance(st.session_state.get("llm_remember_settings", True), bool):
            st.session_state["llm_remember_settings"] = True

        provider = st.selectbox(
            "Provider",
            options=provider_options,
            help="Choose direct providers or openai_compatible for local OpenAI-style endpoints.",
            key="llm_provider",
        )

        openai_api_key = st.text_input("OpenAI API key", type="password", key="openai_api_key")
        anthropic_api_key = st.text_input("Claude API key", type="password", key="anthropic_api_key")
        google_api_key = st.text_input("Gemini API key", type="password", key="google_api_key")

        base_url = st.text_input(
            "OpenAI-compatible base URL",
            key="openai_base_url",
            help="Only used when provider=openai_compatible",
        )
        temperature = st.slider("Temperature", 0.0, 1.2, 0.6, 0.05, key="llm_temperature")

        auto_fetch_models = st.toggle(
            "Auto-fetch models from provider API",
            value=True,
            key="llm_auto_fetch_models",
            disabled=provider == "rule_based",
            help="Fetches current model list dynamically when provider/key changes.",
        )
        verify_ssl = st.toggle(
            "Verify SSL certificates",
            value=True,
            key="llm_verify_ssl",
            disabled=provider == "rule_based",
            help="Disable only if your environment uses custom/intercepting certificates.",
        )
        ca_bundle_path = st.text_input(
            "CA bundle path (optional)",
            key="llm_ca_bundle_path",
            help="Path to PEM bundle for custom certificate chains, e.g. /path/to/cacert.pem",
        )
        remember_settings = st.toggle(
            "Remember LLM settings on this machine",
            value=True,
            key="llm_remember_settings",
            help="Saves provider, model, and API keys to local .sim_ui_settings.json",
        )

        clear_saved = st.button("Forget saved LLM settings")
        if clear_saved:
            clear_ui_settings()
            for key in LLM_SETTING_KEYS:
                if key in st.session_state:
                    if "api_key" in key:
                        st.session_state[key] = ""
            st.session_state["llm_provider"] = "rule_based"
            st.session_state["llm_model_selected"] = "rule_based"
            st.session_state["llm_model_name"] = ""
            st.success("Saved LLM settings cleared.")

        refresh_models = st.button("Refresh model list now", disabled=provider == "rule_based")

        fingerprint = "|".join(
            [
                provider,
                base_url.strip(),
                _hash_secret(openai_api_key),
                _hash_secret(anthropic_api_key),
                _hash_secret(google_api_key),
                str(bool(verify_ssl)),
                ca_bundle_path.strip(),
            ]
        )

        cache = st.session_state.model_catalog_cache
        should_fetch = provider != "rule_based" and (refresh_models or (auto_fetch_models and cache.get("fingerprint") != fingerprint))

        if should_fetch:
            models, error = fetch_models(
                provider=provider,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                google_api_key=google_api_key,
                base_url=base_url,
                ssl_verify=verify_ssl,
                ca_bundle_path=ca_bundle_path,
            )
            cache["fingerprint"] = fingerprint
            cache["models"] = models
            cache["error"] = error or ""
            st.session_state.model_catalog_cache = cache

        fetched_models = cache.get("models", []) if cache.get("fingerprint") == fingerprint else []
        fetch_error = cache.get("error", "") if cache.get("fingerprint") == fingerprint else ""

        presets = fetched_models or default_model_presets(provider)
        options = [*presets, "custom"] if provider != "rule_based" else ["rule_based"]
        if st.session_state.get("llm_model_selected") not in options:
            st.session_state["llm_model_selected"] = options[0]
        selected_preset = st.selectbox("Model", options=options, key="llm_model_selected")

        model_default = "gpt-4o-mini" if provider in {"openai", "openai_compatible"} else ""
        if provider == "rule_based":
            model = "rule_based"
        elif selected_preset == "custom":
            model = st.text_input("Custom model name", value=model_default, key="llm_model_name").strip() or model_default
        else:
            model = selected_preset

        if fetched_models:
            st.caption(f"Fetched {len(fetched_models)} models from {provider} API.")
        elif fetch_error and provider != "rule_based":
            st.warning(f"Dynamic model fetch failed: {fetch_error}")

        if remember_settings:
            save_ui_settings(extract_llm_settings(st.session_state))
            st.caption("LLM settings are persisted to .sim_ui_settings.json")

        st.caption("If provider is rule_based or provider init fails, fallback generation is used.")

    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        google_api_key=google_api_key,
        temperature=temperature,
    )


def render_scenario_panel(state: dict) -> None:
    st.subheader("Scenario Panel")
    scenario = state["scenario"]
    col1, col2 = st.columns(2)

    with col1:
        scenario["name"] = st.text_input("Scenario name", value=scenario.get("name", ""), key="scenario_name")
        scenario["environment_description"] = st.text_area(
            "Environment description",
            value=scenario.get("environment_description", ""),
            key="scenario_env",
            height=120,
        )
        scenario["style_notes"] = st.text_area(
            "Style notes",
            value=scenario.get("style_notes", ""),
            key="scenario_style_notes",
            height=90,
            help="Tone and expression rules (e.g. short, natural, indirect).",
        )
        scenario["safe_word"] = st.text_input("Safe word (optional)", value=scenario.get("safe_word", ""), key="scenario_safeword")

        examples_text = st.text_area(
            "Style examples (one per line)",
            value="\n".join(scenario.get("style_examples", [])),
            key="scenario_style_examples",
            height=120,
            help="Reference lines for speaking style. Not counted as simulation history.",
        )
        scenario["style_examples"] = [line.strip() for line in examples_text.splitlines() if line.strip()][:20]

    with col2:
        scenario["start_stage"] = st.slider(
            "Start stage",
            min_value=1,
            max_value=6,
            value=int(state.get("current_stage", scenario.get("start_stage", 1))),
            key="scenario_start_stage",
        )
        scenario["initial_tension"] = st.slider(
            "Initial tension",
            min_value=0.0,
            max_value=100.0,
            value=float(scenario.get("initial_tension", 25.0)),
            key="scenario_initial_tension",
        )
        scenario["initial_risk"] = st.slider(
            "Initial risk",
            min_value=0.0,
            max_value=100.0,
            value=float(scenario.get("initial_risk", 15.0)),
            key="scenario_initial_risk",
        )
        flags = scenario.setdefault("flags", {})
        flags["substance_effect"] = st.toggle(
            "Substance effect flag",
            value=bool(flags.get("substance_effect", False)),
            key="scenario_substance",
        )


def render_character_panel(state: dict) -> None:
    st.subheader("Character Editor")
    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("Add NPC"):
            char_id = f"npc_{uuid.uuid4().hex[:6]}"
            state["characters"].append(
                {
                    "id": char_id,
                    "name": f"NPC {len(state['characters'])}",
                    "role": "npc",
                    "description": "",
                    "traits": {
                        trait: {
                            "base": 50,
                            "current": 50.0,
                            "volatility": 0.35,
                            "dynamic_enabled": True,
                        }
                        for trait in TRAIT_NAMES
                    },
                }
            )
            st.session_state.sim_state = rebuild_relationships_for_characters(state)
            st.rerun()

    with cols[1]:
        if st.button("Sync Relationships"):
            st.session_state.sim_state = rebuild_relationships_for_characters(state)
            st.rerun()

    with cols[2]:
        st.caption("Slider only controls base. current is runtime-driven by simulation engine.")

    remove_target = None
    for idx, char in enumerate(state["characters"]):
        with st.expander(f"{char['name']} ({char['role']})", expanded=(idx == 0)):
            top = st.columns([2, 1, 1])
            with top[0]:
                char["name"] = st.text_input("Name", value=char["name"], key=f"char_name_{char['id']}")
                char["description"] = st.text_area(
                    "Description", value=char.get("description", ""), key=f"char_desc_{char['id']}", height=68
                )
            with top[1]:
                char["role"] = st.selectbox(
                    "Role",
                    options=["user_controlled", "npc"],
                    index=0 if char.get("role") == "user_controlled" else 1,
                    key=f"char_role_{char['id']}",
                )
                st.text_input("ID", value=char["id"], disabled=True, key=f"char_id_{char['id']}")
            with top[2]:
                if len(state["characters"]) > 1 and st.button("Remove", key=f"remove_{char['id']}"):
                    remove_target = char["id"]

            for trait in TRAIT_NAMES:
                trait_obj = char["traits"][trait]
                tcols = st.columns([2, 2, 2, 2])
                with tcols[0]:
                    trait_obj["base"] = st.slider(
                        f"{trait} base",
                        min_value=0,
                        max_value=100,
                        value=int(trait_obj.get("base", 50)),
                        key=f"trait_base_{char['id']}_{trait}",
                    )
                with tcols[1]:
                    st.number_input(
                        f"{trait} current",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(trait_obj.get("current", trait_obj["base"])),
                        disabled=True,
                        key=f"trait_current_{char['id']}_{trait}",
                    )
                with tcols[2]:
                    trait_obj["volatility"] = st.slider(
                        f"{trait} volatility",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(trait_obj.get("volatility", 0.35)),
                        step=0.01,
                        key=f"trait_vol_{char['id']}_{trait}",
                    )
                with tcols[3]:
                    trait_obj["dynamic_enabled"] = st.toggle(
                        f"{trait} dynamic",
                        value=bool(trait_obj.get("dynamic_enabled", True)),
                        key=f"trait_dyn_{char['id']}_{trait}",
                    )

    if remove_target:
        state["characters"] = [c for c in state["characters"] if c["id"] != remove_target]
        st.session_state.sim_state = rebuild_relationships_for_characters(state)
        st.rerun()


def render_relationship_panel(state: dict) -> None:
    st.subheader("Relationship Matrix")
    chars = {char["id"]: char["name"] for char in state["characters"]}

    if not state["relationships"]:
        st.info("No relationship pair exists. Add at least two characters.")
        return

    for key, rel in state["relationships"].items():
        left_id, right_id = key.split("|")
        title = f"{chars.get(left_id, left_id)} <-> {chars.get(right_id, right_id)}"
        with st.expander(title, expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                rel["trust_level"] = st.slider(
                    "trust_level", 0.0, 100.0, float(rel.get("trust_level", 50.0)), key=f"rel_trust_{key}"
                )
                rel["sexual_tension"] = st.slider(
                    "sexual_tension", 0.0, 100.0, float(rel.get("sexual_tension", 30.0)), key=f"rel_tension_{key}"
                )
            with col2:
                rel["respect_level"] = st.slider(
                    "respect_level", 0.0, 100.0, float(rel.get("respect_level", 50.0)), key=f"rel_respect_{key}"
                )
                rel["dominance_dynamic"] = st.slider(
                    "dominance_dynamic",
                    -50.0,
                    50.0,
                    float(rel.get("dominance_dynamic", 0.0)),
                    key=f"rel_dom_{key}",
                )


def render_simulation_panel(state: dict, llm_cfg: LLMConfig) -> None:
    st.subheader("Simulation Panel")
    stage = int(state["current_stage"])

    st.markdown(
        f"""
        <div class="sim-card">
          <div class="sim-title">Current Stage</div>
          <div class="sim-sub">{stage} - {stage_label(stage)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Tension", f"{state['tension']:.1f}")
    with m2:
        st.metric("Risk", f"{state['risk']:.1f}")
    with m3:
        st.metric("Turn", f"{state.get('turn_index', 0)}")

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])
    with ctrl1:
        state["freeze_stage"] = st.toggle("Freeze Stage", value=bool(state.get("freeze_stage", False)))
    with ctrl2:
        options = [None, 1, 2, 3, 4, 5, 6]
        index = options.index(state.get("manual_stage_override")) if state.get("manual_stage_override") in options else 0
        state["manual_stage_override"] = st.selectbox(
            "Manual Stage Override",
            options=options,
            index=index,
            format_func=lambda x: "None" if x is None else f"{x} - {STAGE_NAMES.get(x, '')}",
        )
    with ctrl3:
        if st.button("Reset Tension & Risk"):
            state["tension"] = float(state["scenario"].get("initial_tension", 25.0))
            state["risk"] = float(state["scenario"].get("initial_risk", 15.0))
            st.rerun()
    with ctrl4:
        if st.button("Reset Whole State"):
            st.session_state.sim_state = create_default_state()
            queue_scenario_widget_state_seed(st.session_state.sim_state)
            st.session_state.last_outputs = []
            st.session_state.last_issues = []
            st.rerun()

    st.markdown("#### Character Runtime Snapshot")
    for char in state["characters"]:
        with st.expander(f"{char['name']} ({char['role']})", expanded=False):
            trait_cols = st.columns(4)
            for idx, trait_name in enumerate(TRAIT_NAMES):
                with trait_cols[idx % 4]:
                    current = float(char["traits"][trait_name].get("current", 50.0))
                    st.progress(int(current), text=f"{trait_name}: {current:.1f}")

    actors = [c for c in state["characters"] if c.get("role") == "user_controlled"] or state["characters"]
    actor_map = {c["name"]: c["id"] for c in actors}
    user_ids = {c["id"] for c in actors}
    id_to_name = {c["id"]: c["name"] for c in state["characters"]}
    id_to_name["scene"] = "Scene"

    npc_count = max(0, len(state["characters"]) - 1)
    state.setdefault("config", {})
    current_npc_responses = int(state.get("config", {}).get("npc_responses_per_turn", min(2, max(1, npc_count))))

    if npc_count <= 1:
        state["config"]["npc_responses_per_turn"] = 1
        st.caption("NPC responses per turn: 1 (only one NPC available)")
    else:
        state["config"]["npc_responses_per_turn"] = st.slider(
            "NPC responses per turn",
            min_value=1,
            max_value=npc_count,
            value=max(1, min(current_npc_responses, npc_count)),
            help="How many NPCs respond automatically after each Mert turn.",
        )
    state["config"]["npc_auto_rounds_on_open"] = st.slider(
        "Auto rounds on scene start",
        min_value=1,
        max_value=6,
        value=int(state.get("config", {}).get("npc_auto_rounds_on_open", 2)),
    )
    state["config"]["npc_auto_rounds_after_user"] = st.slider(
        "Auto rounds after your message",
        min_value=1,
        max_value=6,
        value=int(state.get("config", {}).get("npc_auto_rounds_after_user", 2)),
    )
    state["config"]["npc_auto_rounds_on_advance"] = st.slider(
        "Auto rounds for advance button",
        min_value=1,
        max_value=6,
        value=int(state.get("config", {}).get("npc_auto_rounds_on_advance", 2)),
    )

    llm_probe = LLMClient(llm_cfg)
    if llm_cfg.provider == "rule_based":
        st.caption("LLM provider: rule_based (no API calls)")
    elif llm_probe.ready:
        st.caption(f"LLM provider ready: {llm_cfg.provider} / {llm_cfg.model}")
    else:
        st.warning(
            f"Provider init failed for {llm_cfg.provider}. Fallback generation will be used."
            + (f" Details: {llm_probe.init_error}" if llm_probe.init_error else "")
        )

    st.markdown("#### Dialogue")
    chat_box = st.container(height=430, border=True)
    with chat_box:
        for item in state.get("history", [])[-120:]:
            speaker_id = item.get("speaker", "unknown")
            speaker = id_to_name.get(speaker_id, speaker_id)
            text = item.get("message", "")
            is_user = speaker_id in user_ids
            role = "user" if is_user else "assistant"
            with st.chat_message(role):
                if speaker_id == "scene":
                    st.markdown(f"*{text}*")
                else:
                    st.markdown(f"**{speaker}**\n\n{text}")

    st.markdown("#### Controls")
    selected_name = st.selectbox("Controlled character", options=list(actor_map.keys()))
    control_cols = st.columns([1, 1, 2])
    with control_cols[0]:
        if st.button("Start Simulation (NPC Opening)", type="primary", disabled=bool(state.get("scene_started", False))):
            engine = SimulationEngine(llm_probe)
            state, outputs, issues = engine.start_scene(state, actor_map[selected_name])
            st.session_state.sim_state = normalize_state(state)
            st.session_state.last_outputs = outputs
            st.session_state.last_issues = issues
            st.rerun()
    with control_cols[1]:
        if st.button("Advance Scene (NPC Auto)", disabled=not bool(state.get("scene_started", False))):
            engine = SimulationEngine(llm_probe)
            state, outputs, issues = engine.advance_scene(state, actor_map[selected_name])
            st.session_state.sim_state = normalize_state(state)
            st.session_state.last_outputs = outputs
            st.session_state.last_issues = issues
            st.rerun()
    with control_cols[2]:
        st.caption("When scene starts, NPC side can continue for multiple rounds before and after your turn.")

    user_input = st.chat_input(
        "Mert turnu: mesajini veya aksiyonunu yaz...",
        disabled=not bool(state.get("scene_started", False)),
    )
    if user_input:
        engine = SimulationEngine(llm_probe)
        state, outputs, issues = engine.run_turn(state, actor_map[selected_name], user_input)
        st.session_state.sim_state = normalize_state(state)
        st.session_state.last_outputs = outputs
        st.session_state.last_issues = issues
        st.rerun()

    if st.session_state.last_issues:
        st.markdown("#### Validator Warnings")
        for issue in st.session_state.last_issues:
            st.markdown(
                f'<div class="warning-box"><b>{issue.code}</b>: {issue.message} ({issue.severity})</div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### Validator Log")
    for log in state.get("validator_logs", [])[-10:]:
        if log.get("issues"):
            st.write(
                f"Turn {log['turn']} | Speaker={log.get('speaker', 'n/a')} | Stage {log['stage']} ({log['stage_name']}) | rewrites={log['rewrite_count']} | issues={len(log['issues'])}"
            )


def render_json_panel(state: dict) -> None:
    st.subheader("JSON State")

    c1, c2 = st.columns([2, 1])
    with c1:
        save_path = st.text_input("Save path", value="state_snapshot.json")
    with c2:
        if st.button("Save JSON"):
            path = manager.save(state, save_path)
            st.success(f"Saved to {path}")

    uploaded = st.file_uploader("Load JSON file", type=["json"])
    if uploaded is not None:
        try:
            loaded = manager.load_json_text(uploaded.read().decode("utf-8"))
            st.session_state.sim_state = loaded
            queue_scenario_widget_state_seed(st.session_state.sim_state)
            st.session_state.last_outputs = []
            st.session_state.last_issues = []
            st.success("State loaded from uploaded JSON.")
            st.rerun()
        except Exception as exc:
            st.error(f"Invalid JSON file: {exc}")

    json_text = manager.dump_json(state)
    edited = st.text_area("Edit JSON directly", value=json_text, height=320)
    if st.button("Apply JSON Text"):
        try:
            st.session_state.sim_state = manager.load_json_text(edited)
            queue_scenario_widget_state_seed(st.session_state.sim_state)
            st.session_state.last_outputs = []
            st.session_state.last_issues = []
            st.success("State applied from text.")
            st.rerun()
        except Exception as exc:
            st.error(f"Invalid JSON text: {exc}")


def render_import_panel(state: dict) -> None:
    st.subheader("Import Scenario")
    st.caption("Paste your long-form scenario text. The importer maps it to scenario, characters, relationships, and style history.")

    raw_text = st.text_area(
        "Scenario source text",
        height=360,
        placeholder="Paste your old scenario doc here...",
        key="scenario_import_raw_text",
    )
    chat_to_history = st.toggle(
        "Treat detected chat lines as initial history",
        value=False,
        help="If off, chat logs are used only as style examples unless they appear after an explicit start marker.",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Import (Replace State)", type="primary"):
            imported, summary = import_scenario_text(raw_text, create_default_state(), chat_to_history=chat_to_history)
            st.session_state.sim_state = imported
            queue_scenario_widget_state_seed(st.session_state.sim_state)
            st.session_state.last_outputs = []
            st.session_state.last_issues = []
            st.session_state.import_summary = summary
            st.rerun()

    with c2:
        if st.button("Import (Merge Into Current)"):
            imported, summary = import_scenario_text(raw_text, state, chat_to_history=chat_to_history)
            st.session_state.sim_state = imported
            queue_scenario_widget_state_seed(st.session_state.sim_state)
            st.session_state.last_outputs = []
            st.session_state.last_issues = []
            st.session_state.import_summary = summary
            st.rerun()

    if st.session_state.import_summary:
        info = st.session_state.import_summary
        st.success(
            f"Imported {info.get('characters', 0)} characters, {info.get('history', 0)} history lines, "
            f"{info.get('style_examples', 0)} style examples, "
            f"stage={info.get('stage', '-')}, tension={info.get('tension', '-')}, risk={info.get('risk', '-')}."
        )
        if info.get("safe_word"):
            st.caption(f"Safe word detected: {info['safe_word']}")


def main() -> None:
    st.title("Interactive Character Simulation - Stateful MVP")
    llm_cfg = get_llm_config()

    state = normalize_state(st.session_state.sim_state)
    st.session_state.sim_state = state

    apply_queued_scenario_widget_state_seed()

    if "scenario_name" not in st.session_state:
        queue_scenario_widget_state_seed(state)
        apply_queued_scenario_widget_state_seed()

    tabs = st.tabs(["Scenario", "Characters", "Relationships", "Simulation", "Import", "JSON"])

    with tabs[0]:
        render_scenario_panel(state)
    with tabs[1]:
        render_character_panel(state)
    with tabs[2]:
        render_relationship_panel(state)
    with tabs[3]:
        render_simulation_panel(state, llm_cfg)
    with tabs[4]:
        render_import_panel(state)
    with tabs[5]:
        render_json_panel(state)


if __name__ == "__main__":
    main()
