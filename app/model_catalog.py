from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def default_model_presets(provider: str) -> list[str]:
    presets: dict[str, list[str]] = {
        "openai": ["gpt-4o", "gpt-4.1", "gpt-4o-mini", "o3-mini", "o4-mini"],
        "anthropic": ["claude-3-7-sonnet-latest", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"],
        "gemini": ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
        "openai_compatible": [],
        "rule_based": ["rule_based"],
    }
    return list(presets.get(provider, []))


def _http_get_json(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
    ssl_verify: bool = True,
    ca_bundle_path: str = "",
) -> dict[str, Any]:
    request = urllib.request.Request(url=url, headers=headers or {}, method="GET")
    context = None
    if not ssl_verify:
        context = ssl._create_unverified_context()
    elif ca_bundle_path.strip():
        context = ssl.create_default_context(cafile=ca_bundle_path.strip())

    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        raw = response.read().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("Model API returned non-object JSON payload")
    return payload


def _normalize_openai_models(payload: dict[str, Any]) -> list[str]:
    models: list[str] = []
    for item in payload.get("data", []):
        if isinstance(item, dict):
            model_id = str(item.get("id", "")).strip()
            if model_id:
                models.append(model_id)
    return sorted(set(models))


def _normalize_anthropic_models(payload: dict[str, Any]) -> list[str]:
    models: list[str] = []
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "")).strip()
        if model_id:
            models.append(model_id)
    return sorted(set(models))


def _normalize_gemini_models(payload: dict[str, Any]) -> list[str]:
    models: list[str] = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue

        normalized = name.split("/", 1)[1] if name.startswith("models/") else name
        methods = item.get("supportedGenerationMethods", [])
        if isinstance(methods, list) and methods:
            if "generateContent" not in methods and "generateMessage" not in methods:
                continue

        if "gemini" in normalized.lower():
            models.append(normalized)

    return sorted(set(models))


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _openai_compatible_endpoints(base_url: str) -> list[str]:
    base = _normalize_base_url(base_url)
    if not base:
        return []
    if base.endswith("/v1"):
        return [f"{base}/models"]
    return [f"{base}/models", f"{base}/v1/models"]


def fetch_models(
    provider: str,
    openai_api_key: str = "",
    anthropic_api_key: str = "",
    google_api_key: str = "",
    base_url: str = "",
    ssl_verify: bool = True,
    ca_bundle_path: str = "",
) -> tuple[list[str], str | None]:
    provider = (provider or "").strip()

    if provider == "rule_based":
        return ["rule_based"], None

    try:
        if provider == "openai":
            if not openai_api_key:
                return [], "OpenAI API key required for dynamic model fetch."
            payload = _http_get_json(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {openai_api_key}"},
                ssl_verify=ssl_verify,
                ca_bundle_path=ca_bundle_path,
            )
            models = _normalize_openai_models(payload)
            return models, None

        if provider == "anthropic":
            if not anthropic_api_key:
                return [], "Claude API key required for dynamic model fetch."
            payload = _http_get_json(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                },
                ssl_verify=ssl_verify,
                ca_bundle_path=ca_bundle_path,
            )
            models = _normalize_anthropic_models(payload)
            return models, None

        if provider == "gemini":
            if not google_api_key:
                return [], "Gemini API key required for dynamic model fetch."

            models: list[str] = []
            next_page_token: str | None = None
            max_pages = 4
            page_count = 0
            while page_count < max_pages:
                page_count += 1
                query = {"key": google_api_key}
                if next_page_token:
                    query["pageToken"] = next_page_token
                url = "https://generativelanguage.googleapis.com/v1beta/models?" + urllib.parse.urlencode(query)
                payload = _http_get_json(
                    url,
                    ssl_verify=ssl_verify,
                    ca_bundle_path=ca_bundle_path,
                )
                models.extend(_normalize_gemini_models(payload))
                token = payload.get("nextPageToken")
                next_page_token = str(token).strip() if token else None
                if not next_page_token:
                    break

            return sorted(set(models)), None

        if provider == "openai_compatible":
            endpoints = _openai_compatible_endpoints(base_url)
            if not endpoints:
                return [], "Base URL required for OpenAI-compatible dynamic model fetch."

            auth_headers = {"Authorization": f"Bearer {openai_api_key}"} if openai_api_key else {}
            errors: list[str] = []
            for endpoint in endpoints:
                try:
                    payload = _http_get_json(
                        endpoint,
                        headers=auth_headers,
                        ssl_verify=ssl_verify,
                        ca_bundle_path=ca_bundle_path,
                    )
                    models = _normalize_openai_models(payload)
                    if models:
                        return models, None
                except Exception as exc:
                    errors.append(f"{endpoint}: {exc}")

            return [], " | ".join(errors) if errors else "No models found on OpenAI-compatible endpoint."

        return [], f"Unsupported provider: {provider}"

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        message = detail.strip() or str(exc)
        return [], f"HTTP {exc.code}: {message[:300]}"
    except urllib.error.URLError as exc:
        reason_text = str(exc.reason)
        if "CERTIFICATE_VERIFY_FAILED" in reason_text:
            return [], (
                "Network error: SSL certificate verify failed. "
                "Provide CA bundle path or disable SSL verification for model fetch."
            )
        return [], f"Network error: {reason_text}"
    except Exception as exc:
        return [], str(exc)
