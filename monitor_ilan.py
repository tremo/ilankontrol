#!/usr/bin/env python3
"""Daily monitor for ilan.gov.tr listing pages.

Compares current listing items to previously seen items and sends a Telegram
notification when new entries appear.
"""

from __future__ import annotations

import json
import os
import re
from hashlib import sha1
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.exceptions import SSLError
from urllib3.exceptions import InsecureRequestWarning

TARGET_URL = os.getenv(
    "TARGET_URL",
    "https://www.ilan.gov.tr/ilan/kategori/1/emlak?aci=68&txv=1",
)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
FORCE_TEST_NOTIFICATION = os.getenv("FORCE_TEST_NOTIFICATION", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
STATE_FILE = Path("state.json")
USER_AGENT = "Mozilla/5.0 (compatible; ilan-monitor/1.0)"


@dataclass(frozen=True)
class ListingItem:
    item_id: str
    title: str
    url: str


def fetch_html(url: str) -> str:
    try:
        response = requests.get(
            url,
            timeout=45,
            headers={"User-Agent": USER_AGENT},
            verify=True,
        )
    except SSLError:
        # ilan.gov.tr sometimes serves an incomplete cert chain for some clients.
        requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
        response = requests.get(
            url,
            timeout=45,
            headers={"User-Agent": USER_AGENT},
            verify=False,
        )
    response.raise_for_status()
    return response.text


def clean_text(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip()


def extract_item_id(url: str, title: str) -> str:
    iln_match = re.search(r"(ILN\d{5,})", f"{url} {title}", flags=re.IGNORECASE)
    if iln_match:
        return iln_match.group(1).upper()

    parts = [p for p in urlparse(url).path.split("/") if p]
    if parts:
        tail = parts[-1]
        num_match = re.search(r"(\d{5,})", tail)
        if num_match:
            return num_match.group(1)

    digest = sha1(f"{url}|{title}".encode("utf-8")).hexdigest()[:16]
    return f"HASH:{digest}"


def parse_from_links(html: str, base_url: str) -> Dict[str, ListingItem]:
    soup = BeautifulSoup(html, "html.parser")
    parsed: Dict[str, ListingItem] = {}

    for anchor in soup.select("a[href]"):
        href = (anchor.get("href") or "").strip()
        if not href:
            continue

        absolute_url = urljoin(base_url, href)
        path = urlparse(absolute_url).path

        if "/ilan/" not in path:
            continue
        if path.startswith("/ilan/kategori/"):
            continue

        title = clean_text(anchor.get_text(" ", strip=True))
        if not title:
            title = absolute_url

        item_id = extract_item_id(absolute_url, title)
        parsed[item_id] = ListingItem(item_id=item_id, title=title, url=absolute_url)

    return parsed


def collect_dicts(value: object) -> Iterable[dict]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from collect_dicts(nested)
        return
    if isinstance(value, list):
        for nested in value:
            yield from collect_dicts(nested)


def parse_from_json_ld(html: str, base_url: str) -> Dict[str, ListingItem]:
    soup = BeautifulSoup(html, "html.parser")
    parsed: Dict[str, ListingItem] = {}

    for script in soup.select('script[type="application/ld+json"]'):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        for node in collect_dicts(payload):
            raw_url = node.get("url")
            if not isinstance(raw_url, str):
                continue

            absolute_url = urljoin(base_url, raw_url.strip())
            path = urlparse(absolute_url).path
            if "/ilan/" not in path or path.startswith("/ilan/kategori/"):
                continue

            title = node.get("name")
            if not isinstance(title, str) or not clean_text(title):
                title = absolute_url
            title = clean_text(title)

            item_id = extract_item_id(absolute_url, title)
            parsed[item_id] = ListingItem(item_id=item_id, title=title, url=absolute_url)

    return parsed


def parse_from_regex(html: str, base_url: str) -> Dict[str, ListingItem]:
    parsed: Dict[str, ListingItem] = {}
    url_patterns = [
        r"https?://www\.ilan\.gov\.tr/ilan/\d+/[^\s\"'<>]+",
        r"(?<!kategori)/ilan/\d+/[^\s\"'<>]+",
    ]

    for pattern in url_patterns:
        for raw_url in re.findall(pattern, html, flags=re.IGNORECASE):
            absolute_url = urljoin(base_url, raw_url)
            path = urlparse(absolute_url).path
            if "/ilan/" not in path or path.startswith("/ilan/kategori/"):
                continue

            title = absolute_url
            item_id = extract_item_id(absolute_url, title)
            parsed[item_id] = ListingItem(item_id=item_id, title=title, url=absolute_url)

    return parsed


def parse_iln_fallback(html: str, base_url: str) -> Dict[str, ListingItem]:
    iln_ids = sorted(set(re.findall(r"ILN\d{5,}", html, flags=re.IGNORECASE)))
    fallback: Dict[str, ListingItem] = {}

    for item_id in iln_ids:
        normalized = item_id.upper()
        fallback[normalized] = ListingItem(
            item_id=normalized,
            title=normalized,
            url=base_url,
        )

    return fallback


def parse_items(html: str, base_url: str) -> Dict[str, ListingItem]:
    parsed: Dict[str, ListingItem] = {}
    for parser in (parse_from_links, parse_from_json_ld, parse_from_regex):
        parsed.update(parser(html, base_url))
    if parsed:
        return parsed
    return parse_iln_fallback(html, base_url)


def likely_blocked_response(html: str) -> bool:
    sample = html.lower()
    signatures = [
        "captcha",
        "robot",
        "cloudflare",
        "datadome",
        "forbidden",
        "access denied",
        "erişim engellendi",
        "guvenlik",
    ]
    return any(mark in sample for mark in signatures)


def is_empty_results_page(html: str) -> bool:
    sample = html.lower()
    markers = [
        "kriterlere uygun ilan bulunam",
        "ilan bulunamad",
        "sonuc bulunamad",
        "0 ilan",
    ]
    return any(mark in sample for mark in markers)


def load_state(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, items: Iterable[ListingItem]) -> None:
    payload = {
        item.item_id: {"title": item.title, "url": item.url}
        for item in sorted(items, key=lambda i: i.item_id)
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def send_telegram(message: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError(
            "Telegram credentials missing. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
        )

    api_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    response = requests.post(
        api_url,
        data={"chat_id": CHAT_ID, "text": message},
        timeout=20,
    )
    response.raise_for_status()


def build_message(new_items: list[ListingItem]) -> str:
    lines = [f"Yeni ilan bulundu: {len(new_items)} adet"]
    for item in new_items[:10]:
        lines.append(f"- {item.title}\n{item.url}")
    if len(new_items) > 10:
        lines.append(f"... ve {len(new_items) - 10} adet daha")
    return "\n\n".join(lines)


def main() -> None:
    html = fetch_html(TARGET_URL)
    current_items = parse_items(html, TARGET_URL)
    previous_state = load_state(STATE_FILE)

    if FORCE_TEST_NOTIFICATION and BOT_TOKEN and CHAT_ID:
        sample = list(current_items.values())[:3]
        lines = [f"Test bildirimi: sistem aktif. Parse edilen ilan: {len(current_items)}"]
        for item in sample:
            lines.append(f"- {item.title}\n{item.url}")
        send_telegram("\n\n".join(lines))

    if not current_items:
        if is_empty_results_page(html):
            print(f"Bilgi: hedef sayfada aktif ilan yok ({TARGET_URL}).")
            return

        hint = " Muhtemel neden: bot korumasi/captcha."
        if not likely_blocked_response(html):
            hint = ""
        warning = (
            f"Uyari: hedef sayfadan ilan parse edilemedi ({TARGET_URL}); bu kosu atlandi."
            f"{hint}"
        )
        print(warning)
        if BOT_TOKEN and CHAT_ID:
            try:
                send_telegram(warning)
            except Exception:
                pass
        return

    if not previous_state:
        save_state(STATE_FILE, current_items.values())
        return

    new_items = [
        item for item_id, item in current_items.items() if item_id not in previous_state
    ]
    if new_items:
        send_telegram(build_message(new_items))

    save_state(STATE_FILE, current_items.values())


if __name__ == "__main__":
    main()
