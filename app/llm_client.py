from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMConfig:
    provider: str = "rule_based"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    base_url: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    api_key: str = ""


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._chat_model = None
        self.init_error: str | None = None

        try:
            if config.provider == "openai":
                self._init_openai(compatible=False)
            elif config.provider == "openai_compatible":
                self._init_openai(compatible=True)
            elif config.provider == "anthropic":
                self._init_anthropic()
            elif config.provider == "gemini":
                self._init_gemini()
        except Exception as exc:
            self.init_error = str(exc)
            self._chat_model = None

    @property
    def ready(self) -> bool:
        if self.config.provider == "rule_based":
            return True
        return self._chat_model is not None

    def planner(self, system_prompt: str, prompt: str) -> str:
        if self._chat_model is None:
            return self._fallback_plan(prompt)
        return self._invoke(system_prompt, prompt)

    def generator(self, system_prompt: str, prompt: str) -> str:
        if self._chat_model is None:
            return self._fallback_generation(prompt)
        return self._invoke(system_prompt, prompt)

    def validator(self, system_prompt: str, prompt: str) -> str:
        if self._chat_model is None:
            return "OK"
        return self._invoke(system_prompt, prompt)

    def _invoke(self, system_prompt: str, user_prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = self._chat_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return self._extract_text(resp.content)

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text.strip()
            return str(content).strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item.strip())
                    continue

                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                    continue

                item_text = getattr(item, "text", None)
                if isinstance(item_text, str) and item_text.strip():
                    parts.append(item_text.strip())

            joined = "\n".join(part for part in parts if part)
            return joined.strip()

        return str(content).strip()

    def _init_openai(self, compatible: bool) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing dependency: langchain-openai. Run: pip install langchain-openai") from exc

        key = self.config.openai_api_key or self.config.api_key
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
        }
        if key:
            kwargs["api_key"] = key
        if compatible and self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._chat_model = ChatOpenAI(**kwargs)

    def _init_anthropic(self) -> None:
        try:
            from langchain_anthropic import ChatAnthropic
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing dependency: langchain-anthropic. Run: pip install langchain-anthropic") from exc

        key = self.config.anthropic_api_key or self.config.api_key
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
        }
        if key:
            kwargs["api_key"] = key

        try:
            self._chat_model = ChatAnthropic(**kwargs)
        except TypeError:
            if key:
                kwargs.pop("api_key", None)
                kwargs["anthropic_api_key"] = key
            self._chat_model = ChatAnthropic(**kwargs)

    def _init_gemini(self) -> None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing dependency: langchain-google-genai. Run: pip install langchain-google-genai") from exc

        key = self.config.google_api_key or self.config.api_key
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
        }
        if key:
            kwargs["google_api_key"] = key
        self._chat_model = ChatGoogleGenerativeAI(**kwargs)

    def _fallback_plan(self, prompt: str) -> str:
        styles = [
            "Acknowledge, mirror tone, and keep escalation gradual.",
            "Stay indirect, maintain deniability, and ask one grounded follow-up.",
            "Respond with calm social continuity and minimal forward push.",
        ]
        seed = sum(ord(c) for c in prompt)
        random.seed(seed)
        return random.choice(styles)

    def _fallback_generation(self, prompt: str) -> str:
        low_templates = [
            "I notice the shift and answer in a calm, measured way, keeping things light.",
            "I respond with a small smile, staying conversational and leaving room for interpretation.",
            "I keep my tone steady and engaged, matching the moment without forcing momentum.",
        ]
        medium_templates = [
            "I lean slightly into the moment, hinting at interest while keeping it understated.",
            "I hold eye contact a second longer, then respond with a careful, suggestive note.",
            "I acknowledge the tension indirectly and keep the exchange controlled.",
        ]

        text = prompt.lower()
        if "stage=1" in text or "stage=2" in text:
            table = low_templates
        else:
            table = medium_templates

        seed = sum(ord(c) for c in prompt)
        random.seed(seed)
        return random.choice(table)
