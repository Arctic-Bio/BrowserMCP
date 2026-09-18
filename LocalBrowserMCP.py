#!/usr/bin/env python3
"""
browser_mcp.py — local Playwright browser-control MCP server.

A single-file MCP server intended for local AI agents. It exposes a persistent
Chromium profile plus a compact, intent-oriented page snapshot:

    action     clickable controls, buttons, toggles, checkboxes, etc.
    link       hyperlinks
    input      text fields, textareas, selects, file inputs
    title      headings
    visual     images, SVG, canvas, video
    container  semantic regions such as nav/main/dialog/form
    text       meaningful visible text

Important design goals:
    - Works with the current MCP Python SDK (v2) and can fall back to v1.
    - Stable element references survive normal DOM re-renders on the same page.
    - Automatically invalidates stale references after navigation.
    - Persistent browser login/profile.
    - Useful diagnostics: console log, page errors, page info.
    - Safer browser lifecycle and tab bookkeeping.
    - More useful interaction primitives: hover, select, upload, fill, wait.
    - Verbose structured responses so an agent can recover from failures.

Install in a venv:
    python -m pip install -U "mcp[cli]" playwright
    python -m playwright install chromium

Run:
    python browser_mcp.py

For an MCP client using stdio, point it at this file. Do not print ordinary
logging to stdout: stdout belongs to the MCP protocol. Human diagnostics go
to stderr.

Environment variables:
    BROWSER_MCP_PROFILE_DIR   Persistent Chromium profile directory.
    BROWSER_MCP_HEADLESS      1 for headless, 0 for headed (default).
    BROWSER_MCP_VIEWPORT_W    Viewport width (default 1440).
    BROWSER_MCP_VIEWPORT_H    Viewport height (default 900).
    BROWSER_MCP_SETTLE_MS      Post-action settle delay (default 300).
    BROWSER_MCP_TIMEOUT_MS     Playwright action/navigation timeout (default 15s).
    BROWSER_MCP_MAX_TEXT_NODES Maximum text nodes in a snapshot (default 150).
    BROWSER_MCP_MAX_CONSOLE    Console ring-buffer size (default 500).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal, Optional

# ---------------------------------------------------------------------------
# MCP SDK compatibility
# ---------------------------------------------------------------------------
#
# Current official SDK v2 uses MCPServer. Older v1 uses FastMCP. The everyday
# decorator API is intentionally very similar, so keeping a small import shim
# is worthwhile for machines that have not upgraded yet.
#
# Current v2 documentation:
# https://py.sdk.modelcontextprotocol.io/
#
try:
    from mcp.server.mcpserver import MCPServer  # MCP SDK v2
    FastMCP = MCPServer
    MCP_SDK_MAJOR = 2
    try:
        from mcp.server.mcpserver import Image
    except ImportError:
        from mcp.server.mcpserver.utilities.types import Image
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP  # MCP SDK v1
        MCP_SDK_MAJOR = 1
        try:
            from mcp.server.fastmcp import Image
        except ImportError:
            from mcp.server.fastmcp.utilities.types import Image
    except ImportError as exc:
        raise RuntimeError(
            "The MCP Python SDK is not installed. Install it with:\n"
            '  python -m pip install -U "mcp[cli]"'
        ) from exc

from playwright.async_api import (
    BrowserContext,
    Locator,
    Page,
    Playwright,
    TimeoutError as PWTimeoutError,
    async_playwright,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CONSOLE_LOG = 500

DEFAULT_EVALUATE_TEXT_MAX_CHARS = 10000

MAX_EVALUATE_TEXT_MAX_CHARS = 100000


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


PROFILE_DIR = Path(
    os.environ.get(
        "BROWSER_MCP_PROFILE_DIR",
        str(Path.home() / ".browser_mcp_profile"),
    )
).expanduser()

HEADLESS_DEFAULT = os.environ.get("BROWSER_MCP_HEADLESS", "0") == "1"
DEFAULT_VIEWPORT = {
    "width": _env_int("BROWSER_MCP_VIEWPORT_W", 1440, 320),
    "height": _env_int("BROWSER_MCP_VIEWPORT_H", 900, 240),
}
MAX_TEXT_NODES = _env_int("BROWSER_MCP_MAX_TEXT_NODES", 150, 1)
LABEL_MAX_LEN = 200
NAV_SETTLE_MS = _env_int("BROWSER_MCP_SETTLE_MS", 300, 0)
ACTION_TIMEOUT_MS = _env_int("BROWSER_MCP_TIMEOUT_MS", 15_000, 500)
MAX_CONSOLE_LOG = _env_int("BROWSER_MCP_MAX_CONSOLE", 500, 10)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(message: str) -> None:
    """Write diagnostics to stderr, never stdout."""
    print(f"[browser-mcp] {message}", file=sys.stderr, flush=True)


def ok(message: str, **data: Any) -> str:
    payload = {"ok": True, "message": message}
    payload.update(data)
    return json.dumps(payload, ensure_ascii=False, default=str)


def fail(message: str, **data: Any) -> str:
    payload = {"ok": False, "error": message}
    payload.update(data)
    return json.dumps(payload, ensure_ascii=False, default=str)


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty.")
    if not url.startswith(("http://", "https://", "file://", "about:", "data:")):
        url = "https://" + url
    return url


# ---------------------------------------------------------------------------
# Page extraction script
# ---------------------------------------------------------------------------

# This deliberately returns a FLAT representation. Flat records are easier
# for an LLM to filter, search, diff, and act upon than a huge DOM tree.
EXTRACTION_JS = r"""
() => {
    if (!window.__mcpState) {
        window.__mcpState = {
            counter: 0,
            refMap: new WeakMap(),
            generation: 0
        };
    }

    const state = window.__mcpState;

    function getRef(el) {
        if (state.refMap.has(el)) return state.refMap.get(el);

        state.counter += 1;
        const ref = 'e' + state.counter;
        state.refMap.set(el, ref);
        el.setAttribute('data-mcp-ref', ref);
        return ref;
    }

    function isVisible(el) {
        if (!el || !el.isConnected) return false;
        if (el.hidden) return false;
        if (el.getAttribute('aria-hidden') === 'true') return false;

        const style = window.getComputedStyle(el);
        if (style.display === 'none' ||
            style.visibility === 'hidden' ||
            parseFloat(style.opacity || '1') === 0) {
            return false;
        }

        const rect = el.getBoundingClientRect();
        return rect.width > 0 || rect.height > 0;
    }

    function cleanText(value, maxLen = 200) {
        return (value || '')
            .replace(/\s+/g, ' ')
            .trim()
            .slice(0, maxLen);
    }

    function accessibleName(el) {
        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel) return cleanText(ariaLabel);

        const labelledBy = el.getAttribute('aria-labelledby');
        if (labelledBy) {
            const parts = labelledBy
                .split(/\s+/)
                .map(id => document.getElementById(id))
                .filter(Boolean)
                .map(node => node.innerText || node.textContent || '');
            const value = cleanText(parts.join(' '));
            if (value) return value;
        }

        const title = el.getAttribute('title');
        if (title) return cleanText(title);

        const alt = el.getAttribute('alt');
        if (alt) return cleanText(alt);

        const placeholder = el.getAttribute('placeholder');
        if (placeholder) return cleanText(placeholder);

        const tag = el.tagName.toLowerCase();
        if (tag === 'input' || tag === 'textarea' || tag === 'select') {
            const value = el.value;
            if (value) return cleanText(value);
        }

        // Labels explicitly associated with a form control.
        if (el.id) {
            const label = document.querySelector(
                `label[for="${CSS.escape(el.id)}"]`
            );
            if (label) {
                const value = cleanText(label.innerText || label.textContent);
                if (value) return value;
            }
        }

        return cleanText(el.innerText || el.textContent || '');
    }

    const LANDMARK_SELECTOR = [
        'nav', 'header', 'footer', 'main', 'form', 'dialog', 'aside',
        '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]',
        '[role="main"]', '[role="dialog"]', '[role="form"]',
        '[role="region"]', 'section[aria-label]'
    ].join(',');

    function nearestRegion(el) {
        let cur = el.parentElement;
        let hops = 0;
        while (cur && hops < 40) {
            if (cur.matches && cur.matches(LANDMARK_SELECTOR)) {
                const label =
                    cur.getAttribute('aria-label') ||
                    cur.getAttribute('aria-labelledby');
                if (label) return cleanText(label, 60);
                return cur.tagName.toLowerCase();
            }
            cur = cur.parentElement;
            hops += 1;
        }
        return 'body';
    }

    function classify(el) {
        const tag = el.tagName.toLowerCase();
        const role = (el.getAttribute('role') || '').toLowerCase();
        const type = (el.getAttribute('type') || '').toLowerCase();
        const hasHandler =
            !!el.onclick || !!el.getAttribute('onclick');

        if (['textarea', 'select'].includes(tag)) return 'input';

        if (
            tag === 'input' &&
            ![
                'button', 'submit', 'checkbox', 'radio',
                'image', 'reset', 'file'
            ].includes(type)
        ) {
            return 'input';
        }

        if (tag === 'input' && type === 'file') return 'input';
        if (tag === 'input' && ['checkbox', 'radio'].includes(type)) {
            return 'action';
        }

        if (
            tag === 'button' ||
            role === 'button' ||
            type === 'submit' ||
            type === 'button' ||
            hasHandler
        ) {
            return 'action';
        }

        if (
            [
                'checkbox', 'radio', 'switch', 'tab',
                'menuitem', 'option', 'combobox'
            ].includes(role)
        ) {
            return role === 'combobox' ? 'input' : 'action';
        }

        if (tag === 'a' && el.hasAttribute('href')) return 'link';

        if (/^h[1-6]$/.test(tag) || role === 'heading') return 'title';

        if (['img', 'svg', 'canvas', 'video', 'picture'].includes(tag)) {
            return 'visual';
        }

        const landmarkTags = [
            'nav', 'header', 'footer', 'main',
            'form', 'dialog', 'aside'
        ];

        if (
            landmarkTags.includes(tag) ||
            [
                'navigation', 'banner', 'contentinfo',
                'main', 'dialog', 'region'
            ].includes(role)
        ) {
            return 'container';
        }

        return 'text';
    }

    const SELECTOR = [
        'a[href]',
        'button',
        '[role="button"]',
        '[onclick]',
        'input',
        'textarea',
        'select',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        '[role="heading"]',
        'img',
        'svg[aria-label]',
        'video',
        'canvas',
        'nav', 'header', 'footer', 'main', 'form', 'dialog', 'aside',
        '[role="navigation"]', '[role="banner"]', '[role="dialog"]',
        '[role="main"]', '[role="region"]',
        'p', 'li', 'label', 'td', 'th',
        'span[class*="price"]',
        '[role="checkbox"]', '[role="radio"]',
        '[role="switch"]', '[role="tab"]',
        '[role="menuitem"]', '[role="option"]',
        '[role="combobox"]'
    ].join(',');

    const results = [];
    const seen = new Set();
    let textCount = 0;

    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const sx = window.scrollX;
    const sy = window.scrollY;

    for (const el of document.querySelectorAll(SELECTOR)) {
        if (seen.has(el)) continue;
        seen.add(el);

        if (!isVisible(el)) continue;

        const category = classify(el);

        if (category === 'text') {
            const txt = cleanText(el.innerText || el.textContent || '', 400);
            if (txt.length < 2 || txt.length > 300) continue;

            const onlyChild =
                el.children.length === 1 ? el.children[0] : null;

            if (
                onlyChild &&
                cleanText(onlyChild.innerText || onlyChild.textContent) ===
                    cleanText(el.innerText || el.textContent) &&
                onlyChild.matches(SELECTOR)
            ) {
                continue;
            }

            textCount += 1;
            if (textCount > __MAX_TEXT_NODES__) continue;
        }

        const rect = el.getBoundingClientRect();
        const ref = getRef(el);

        let label = '';
        if (category === 'container') {
            label = cleanText(el.getAttribute('aria-label') || '', 80);
            if (!label) {
                const heading = el.querySelector(
                    'h1,h2,h3,h4,h5,h6,[role="heading"]'
                );
                if (heading) {
                    label = cleanText(
                        heading.innerText || heading.textContent || '',
                        80
                    );
                }
            }
        } else {
            label = accessibleName(el);
            if (!label && category === 'visual') {
                label = '(unlabeled ' + el.tagName.toLowerCase() + ')';
            }
        }

        const entry = {
            ref,
            tag: category,
            html_tag: el.tagName.toLowerCase(),
            label,
            region: nearestRegion(el),
            bbox: [
                Math.round(rect.left),
                Math.round(rect.top),
                Math.round(rect.width),
                Math.round(rect.height)
            ],
            in_viewport:
                rect.bottom > 0 &&
                rect.top < vh &&
                rect.right > 0 &&
                rect.left < vw
        };

        if (category === 'link') {
            entry.href = el.href || el.getAttribute('href');
            entry.target = el.getAttribute('target') || null;
        }

        if (category === 'title') {
            const match = el.tagName.toLowerCase().match(/^h([1-6])$/);
            entry.level =
                match
                    ? parseInt(match[1], 10)
                    : (el.getAttribute('aria-level') || null);
        }

        if (category === 'input') {
            const tag = el.tagName.toLowerCase();
            entry.input_type =
                tag === 'input'
                    ? (el.getAttribute('type') || 'text')
                    : tag;

            entry.value =
                typeof el.value === 'string'
                    ? el.value.slice(0, 200)
                    : '';

            entry.required = !!el.required ||
                el.getAttribute('aria-required') === 'true';

            entry.disabled = !!el.disabled ||
                el.getAttribute('aria-disabled') === 'true';
        }

        if (category === 'action') {
            entry.disabled = !!el.disabled ||
                el.getAttribute('aria-disabled') === 'true';

            const role = (el.getAttribute('role') || '').toLowerCase();
            const inputType =
                (el.getAttribute('type') || '').toLowerCase();

            if (
                ['checkbox', 'radio'].includes(inputType) ||
                ['checkbox', 'radio', 'switch'].includes(role)
            ) {
                entry.checked =
                    !!el.checked ||
                    el.getAttribute('aria-checked') === 'true';
            }
        }

        results.push(entry);
    }

    return {
        url: window.location.href,
        title: document.title,
        scroll: {
            x: sx,
            y: sy,
            viewport_width: vw,
            viewport_height: vh,
            page_height: document.documentElement.scrollHeight
        },
        elements: results
    };
}
""".replace("__MAX_TEXT_NODES__", str(MAX_TEXT_NODES))


# ---------------------------------------------------------------------------
# Browser session
# ---------------------------------------------------------------------------

class BrowserSession:
    """Owns Playwright, the persistent context, pages, and snapshot caches."""

    def __init__(self) -> None:
        self._pw: Optional[Playwright] = None
        self.context: Optional[BrowserContext] = None
        self.pages: list[Page] = []
        self.active_index = 0

        # Per-page caches.
        self._last_snapshot: dict[int, dict[str, dict]] = {}
        self._console_log: dict[int, list[dict]] = {}
        self._page_urls: dict[int, str] = {}

        self._lock = asyncio.Lock()

    async def ensure_started(self, headless: bool = HEADLESS_DEFAULT) -> None:
        async with self._lock:
            if self.context is not None:
                return

            PROFILE_DIR.mkdir(parents=True, exist_ok=True)

            log(
                f"Starting Chromium "
                f"(headless={headless}, profile={PROFILE_DIR})"
            )

            self._pw = await async_playwright().start()

            self.context = await self._pw.chromium.launch_persistent_context(
                user_data_dir=str(PROFILE_DIR),
                headless=headless,
                viewport=DEFAULT_VIEWPORT,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-background-networking",
                    "--disable-background-timer-throttling",
                    "--disable-renderer-backgrounding",
                ],
            )

            self.context.set_default_timeout(ACTION_TIMEOUT_MS)
            self.context.set_default_navigation_timeout(ACTION_TIMEOUT_MS)

            try:
                await self.context.grant_permissions(
                    ["clipboard-read", "clipboard-write"]
                )
            except Exception as exc:
                log(f"Clipboard permissions unavailable: {exc}")

            self.context.on("page", self._on_new_page)

            self.pages = list(self.context.pages)

            if not self.pages:
                self.pages = [await self.context.new_page()]

            for page in list(self.pages):
                self._attach_page(page)

            self.active_index = 0

    def _attach_page(self, page: Page) -> None:
        if page not in self.pages:
            self.pages.append(page)

        key = id(page)
        self._console_log.setdefault(key, [])
        self._page_urls.setdefault(key, page.url)

        # Avoid attaching duplicate listeners.
        if getattr(page, "_browser_mcp_attached", False):
            return

        setattr(page, "_browser_mcp_attached", True)

        def on_console(msg: Any) -> None:
            buf = self._console_log.setdefault(key, [])
            buf.append(
                {
                    "type": msg.type,
                    "text": msg.text,
                    "time": time.time(),
                }
            )
            if len(buf) > MAX_CONSOLE_LOG:
                del buf[: len(buf) - MAX_CONSOLE_LOG]

        def on_pageerror(exc: Any) -> None:
            buf = self._console_log.setdefault(key, [])
            buf.append(
                {
                    "type": "pageerror",
                    "text": str(exc),
                    "time": time.time(),
                }
            )
            if len(buf) > MAX_CONSOLE_LOG:
                del buf[: len(buf) - MAX_CONSOLE_LOG]

        def on_close(_: Any) -> None:
            self._drop_page_cache(page)

        page.on("console", on_console)
        page.on("pageerror", on_pageerror)
        page.on("close", on_close)

    def _on_new_page(self, page: Page) -> None:
        self._attach_page(page)
        log(f"New browser tab detected: {page.url}")

    def _drop_page_cache(self, page: Page) -> None:
        key = id(page)
        self._last_snapshot.pop(key, None)
        self._console_log.pop(key, None)
        self._page_urls.pop(key, None)

        if page in self.pages:
            index = self.pages.index(page)
            self.pages.pop(index)
            if self.pages:
                self.active_index = min(self.active_index, len(self.pages) - 1)
            else:
                self.active_index = 0

    def _sync_pages(self) -> None:
        if self.context is None:
            return
        live = list(self.context.pages)

        # Preserve Playwright's actual page list, removing closed tabs.
        self.pages = [p for p in self.pages if not p.is_closed() and p in live]
        for page in live:
            self._attach_page(page)

        if self.pages:
            self.active_index = min(self.active_index, len(self.pages) - 1)

    @property
    def page(self) -> Page:
        self._sync_pages()
        if not self.pages:
            raise RuntimeError("No browser pages are open.")
        return self.pages[self.active_index]

    async def close(self) -> None:
        async with self._lock:
            if self.context is not None:
                try:
                    await self.context.close()
                finally:
                    self.context = None

            if self._pw is not None:
                try:
                    await self._pw.stop()
                finally:
                    self._pw = None

            self.pages.clear()
            self._last_snapshot.clear()
            self._console_log.clear()
            self._page_urls.clear()
            self.active_index = 0

    async def locate(self, ref: str) -> Locator:
        ref = ref.strip()
        if not re.fullmatch(r"e\d+", ref):
            raise ValueError(
                f"Invalid element ref {ref!r}. Expected something like 'e14'."
            )

        page = self.page
        loc = page.locator(f'[data-mcp-ref="{ref}"]').first

        try:
            count = await loc.count()
        except Exception as exc:
            raise RuntimeError(f"Could not inspect ref {ref}: {exc}") from exc

        if count == 0:
            raise ValueError(
                f"No element found for ref '{ref}'. "
                "The page may have changed or the ref may be stale. "
                "Call snapshot(mode='full') again."
            )

        return loc

    async def extract(self) -> dict:
        page = self.page

        # A navigation can leave the page in a transient state. A short settle
        # is intentionally used instead of a heavyweight network-idle wait,
        # because SPAs may keep network requests open forever.
        if NAV_SETTLE_MS:
            await page.wait_for_timeout(NAV_SETTLE_MS)

        try:
            raw = await page.evaluate(EXTRACTION_JS)
        except Exception as exc:
            raise RuntimeError(
                f"Could not extract page elements: {exc}"
            ) from exc

        current_url = raw.get("url", page.url)
        key = id(page)
        previous_url = self._page_urls.get(key)

        if previous_url and previous_url != current_url:
            # DOM refs belong to a document. Never pretend an e7 from the old
            # document still means the same thing after a real navigation.
            self._last_snapshot[key] = {}

        self._page_urls[key] = current_url
        return raw

    def diff_against_last(
        self,
        page_key: int,
        elements: list[dict],
    ) -> dict:
        current = {el["ref"]: el for el in elements}
        previous = self._last_snapshot.get(page_key, {})

        added: list[dict] = []
        changed: list[dict] = []
        removed: list[str] = []

        for ref, element in current.items():
            if ref not in previous:
                added.append(element)
            elif previous[ref] != element:
                changed.append(element)

        for ref in previous:
            if ref not in current:
                removed.append(ref)

        self._last_snapshot[page_key] = current

        return {
            "added": added,
            "changed": changed,
            "removed": removed,
            "total_elements": len(current),
        }

    def full_snapshot_cache(self, page_key: int) -> dict[str, dict]:
        return self._last_snapshot.get(page_key, {})

    def console_log(self, page_key: int) -> list[dict]:
        return self._console_log.get(page_key, [])


SESSION = BrowserSession()


# ---------------------------------------------------------------------------
# Formatting / common helpers
# ---------------------------------------------------------------------------

def _fmt_elements(
    elements: list[dict],
    limit: Optional[int] = None,
) -> list[dict]:
    out = []

    selected = elements[:limit] if limit else elements

    for el in selected:
        item = {
            "ref": el["ref"],
            "tag": el["tag"],
            "label": el.get("label", ""),
            "region": el.get("region", "body"),
        }

        for extra in (
            "html_tag",
            "href",
            "target",
            "level",
            "input_type",
            "value",
            "required",
            "disabled",
            "checked",
            "in_viewport",
            "bbox",
        ):
            if extra in el:
                item[extra] = el[extra]

        out.append(item)

    return out


async def _settle(page: Page, ms: Optional[int] = None) -> None:
    delay = NAV_SETTLE_MS if ms is None else max(0, ms)
    if delay:
        await page.wait_for_timeout(delay)


async def _safe_goto(page: Page, url: str) -> None:
    try:
        await page.goto(url, wait_until="domcontentloaded")
    except PWTimeoutError as exc:
        # A navigation timeout does not necessarily mean navigation failed.
        # Modern sites can remain busy long after DOMContentLoaded.
        log(f"Navigation timeout for {url}: {exc}")
        if not page.url:
            raise
    await _settle(page)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "browser-mcp",
    instructions=(
        "Local Playwright browser-control server. "
        "Use snapshot() to inspect the page and its eN element references "
        "before interacting. Re-snapshot after navigation or major DOM changes."
    ),
)


@mcp.tool()
async def start_browser(headless: bool = HEADLESS_DEFAULT) -> str:
    """Start the persistent Chromium session."""
    try:
        await SESSION.ensure_started(headless=headless)
        return ok(
            "Browser started.",
            profile=str(PROFILE_DIR),
            headless=headless,
            tabs=len(SESSION.pages),
            active_tab=SESSION.active_index,
            url=SESSION.page.url,
        )
    except Exception as exc:
        log(f"start_browser failed: {exc}")
        return fail(
            "Could not start Chromium. Check that Playwright Chromium is installed.",
            detail=str(exc),
        )


@mcp.tool()
async def navigate(url: str) -> str:
    """Navigate the active tab to a URL."""
    try:
        await SESSION.ensure_started()
        normalized = normalize_url(url)
        old_url = SESSION.page.url
        await _safe_goto(SESSION.page, normalized)

        return ok(
            "Navigation completed.",
            requested_url=normalized,
            previous_url=old_url,
            current_url=SESSION.page.url,
            title=await SESSION.page.title(),
            note="Take a fresh snapshot before using element refs from this page.",
        )
    except Exception as exc:
        return fail("Navigation failed.", url=url, detail=str(exc))


@mcp.tool()
async def snapshot(
    mode: Literal["full", "diff"] = "diff",
    viewport_only: bool = False,
    include_visuals: bool = True,
) -> str:
    """
    Inspect the current page using compact semantic element records.

    mode='full' returns all captured elements.
    mode='diff' returns only elements added/changed/removed since the previous
    snapshot for this page.

    viewport_only keeps only currently visible-on-screen elements.
    include_visuals=False removes images/SVG/video/canvas records to reduce
    token usage.
    """
    try:
        await SESSION.ensure_started()

        if mode not in ("full", "diff"):
            return fail("mode must be 'full' or 'diff'.")

        raw = await SESSION.extract()
        elements = raw["elements"]

        if not include_visuals:
            elements = [e for e in elements if e["tag"] != "visual"]

        if viewport_only:
            elements = [e for e in elements if e.get("in_viewport")]

        page_key = id(SESSION.page)

        if mode == "full":
            SESSION._last_snapshot[page_key] = {
                e["ref"]: e for e in elements
            }
            payload = {
                "url": raw["url"],
                "title": raw["title"],
                "scroll": raw["scroll"],
                "mode": "full",
                "elements": _fmt_elements(elements),
                "hint": (
                    "Element refs are document-local. "
                    "After navigation or a large DOM change, snapshot again."
                ),
            }
        else:
            diff = SESSION.diff_against_last(page_key, elements)
            payload = {
                "url": raw["url"],
                "title": raw["title"],
                "scroll": raw["scroll"],
                "mode": "diff",
                "added": _fmt_elements(diff["added"]),
                "changed": _fmt_elements(diff["changed"]),
                "removed_refs": diff["removed"],
                "total_elements": diff["total_elements"],
                "hint": (
                    "If a requested ref is missing, take snapshot(mode='full')."
                ),
            }

        return json.dumps(payload, ensure_ascii=False, default=str)

    except Exception as exc:
        return fail("Snapshot failed.", detail=str(exc))


@mcp.tool()
async def search_elements(
    tag: Optional[str] = None,
    contains: Optional[str] = None,
    viewport_only: bool = False,
    limit: int = 100,
) -> str:
    """
    Search the most recent snapshot cache without touching the browser.

    tag may be action/link/input/title/visual/container/text.
    contains performs a case-insensitive label search.
    """
    try:
        if not SESSION.pages:
            return fail("No browser tab is open.")

        if tag and tag not in {
            "action", "link", "input", "title",
            "visual", "container", "text",
        }:
            return fail("Unknown tag.", valid_tags=[
                "action", "link", "input", "title",
                "visual", "container", "text",
            ])

        page_key = id(SESSION.page)
        cache = SESSION.full_snapshot_cache(page_key)

        if not cache:
            return fail(
                "No snapshot cached yet. Call snapshot(mode='full') first."
            )

        results = list(cache.values())

        if tag:
            results = [e for e in results if e["tag"] == tag]

        if contains:
            needle = contains.casefold()
            results = [
                e for e in results
                if needle in e.get("label", "").casefold()
            ]

        if viewport_only:
            results = [e for e in results if e.get("in_viewport")]

        limit = max(1, min(limit, 1000))

        return ok(
            "Element search completed.",
            count=min(len(results), limit),
            results=_fmt_elements(results, limit),
        )

    except Exception as exc:
        return fail("Element search failed.", detail=str(exc))


@mcp.tool()
async def click(
    ref: str,
    button: Literal["left", "right", "middle"] = "left",
    double: bool = False,
    force: bool = False,
) -> str:
    """Click an element by snapshot ref."""
    try:
        loc = await SESSION.locate(ref)
        await loc.scroll_into_view_if_needed()

        if double:
            await loc.dblclick(button=button, force=force)
        else:
            await loc.click(button=button, force=force)

        await _settle(SESSION.page)

        return ok(
            "Click completed.",
            ref=ref,
            double=double,
            current_url=SESSION.page.url,
            note="Re-snapshot if the page changed.",
        )
    except Exception as exc:
        return fail(
            "Click failed.",
            ref=ref,
            detail=str(exc),
            recovery="Take a fresh snapshot and verify the ref.",
        )


@mcp.tool()
async def multi_click(
    ref: Optional[str] = None,
    count: int = 2,
    delay_ms: int = 250,
    button: Literal["left", "right", "middle"] = "left",
    sequence: Optional[list[dict]] = None,
) -> str:
    """
    Repeat one target or execute a timed click sequence.

    sequence items:
      {"ref": "e12", "delay_ms": 0, "button": "left"}
    """
    try:
        results: list[dict] = []

        if sequence:
            if len(sequence) > 100:
                return fail("Click sequence is limited to 100 steps.")

            for i, step in enumerate(sequence):
                step_ref = str(step.get("ref", "")).strip()
                if not step_ref:
                    results.append({
                        "step": i,
                        "ok": False,
                        "error": "missing ref",
                    })
                    continue

                wait_ms = max(0, int(step.get("delay_ms", 0) or 0))
                step_button = step.get("button", "left")

                if wait_ms:
                    await asyncio.sleep(wait_ms / 1000)

                try:
                    loc = await SESSION.locate(step_ref)
                    await loc.scroll_into_view_if_needed()
                    await loc.click(button=step_button)
                    results.append({
                        "step": i,
                        "ok": True,
                        "ref": step_ref,
                    })
                except Exception as exc:
                    results.append({
                        "step": i,
                        "ok": False,
                        "ref": step_ref,
                        "error": str(exc),
                    })

            await _settle(SESSION.page)
            return ok("Click sequence completed.", results=results)

        if not ref:
            return fail(
                "Provide either ref for repeated clicking or sequence."
            )

        count = max(1, min(int(count), 1000))
        delay_ms = max(0, min(int(delay_ms), 30_000))

        for i in range(count):
            try:
                loc = await SESSION.locate(ref)
                await loc.scroll_into_view_if_needed()
                await loc.click(button=button)
                results.append({
                    "click": i + 1,
                    "ok": True,
                    "ref": ref,
                })
            except Exception as exc:
                results.append({
                    "click": i + 1,
                    "ok": False,
                    "ref": ref,
                    "error": str(exc),
                })
                break

            if i < count - 1 and delay_ms:
                await asyncio.sleep(delay_ms / 1000)

        await _settle(SESSION.page)
        return ok("Repeated clicking completed.", results=results)

    except Exception as exc:
        return fail("multi_click failed.", detail=str(exc))


@mcp.tool()
async def type_text(
    ref: str,
    text: str,
    clear_first: bool = True,
    press_enter: bool = False,
    delay_ms: int = 15,
) -> str:
    """Fill/type text into an input or textarea."""
    try:
        loc = await SESSION.locate(ref)
        await loc.scroll_into_view_if_needed()
        await loc.click()

        if clear_first:
            await loc.fill("")
            await loc.type(text, delay=max(0, min(delay_ms, 1000)))
        else:
            await loc.type(text, delay=max(0, min(delay_ms, 1000)))

        if press_enter:
            await loc.press("Enter")
            await _settle(SESSION.page)

        return ok(
            "Text entry completed.",
            ref=ref,
            characters=len(text),
            pressed_enter=press_enter,
        )
    except Exception as exc:
        return fail(
            "Text entry failed.",
            ref=ref,
            detail=str(exc),
        )


@mcp.tool()
async def fill(ref: str, text: str, press_enter: bool = False) -> str:
    """Set a form control's value directly, normally faster than simulated typing."""
    try:
        loc = await SESSION.locate(ref)
        await loc.scroll_into_view_if_needed()
        await loc.fill(text)

        if press_enter:
            await loc.press("Enter")
            await _settle(SESSION.page)

        return ok(
            "Field filled.",
            ref=ref,
            characters=len(text),
            pressed_enter=press_enter,
        )
    except Exception as exc:
        return fail("Fill failed.", ref=ref, detail=str(exc))


@mcp.tool()
async def select_option(
    ref: str,
    value: Optional[str] = None,
    label: Optional[str] = None,
    index: Optional[int] = None,
) -> str:
    """Select an option in a <select> control by value, label, or index."""
    supplied = sum(x is not None for x in (value, label, index))
    if supplied != 1:
        return fail("Provide exactly one of value, label, or index.")

    try:
        loc = await SESSION.locate(ref)

        if value is not None:
            selected = await loc.select_option(value=value)
        elif label is not None:
            selected = await loc.select_option(label=label)
        else:
            selected = await loc.select_option(index=int(index))

        await _settle(SESSION.page)

        return ok(
            "Option selected.",
            ref=ref,
            selected=selected,
        )
    except Exception as exc:
        return fail("select_option failed.", ref=ref, detail=str(exc))


@mcp.tool()
async def upload_file(ref: str, file_paths: list[str]) -> str:
    """Set one or more files on an <input type=file> control."""
    try:
        if not file_paths:
            return fail("file_paths cannot be empty.")

        normalized = []
        for raw in file_paths:
            path = Path(raw).expanduser().resolve()
            if not path.is_file():
                return fail("File does not exist.", path=str(path))
            normalized.append(str(path))

        loc = await SESSION.locate(ref)
        await loc.set_input_files(normalized)

        return ok(
            "File upload control populated.",
            ref=ref,
            files=normalized,
        )
    except Exception as exc:
        return fail("File upload failed.", ref=ref, detail=str(exc))


@mcp.tool()
async def press_key(key: str) -> str:
    """Press a keyboard key on the active page."""
    try:
        await SESSION.page.keyboard.press(key)
        return ok("Key pressed.", key=key)
    except Exception as exc:
        return fail("Keyboard action failed.", key=key, detail=str(exc))


@mcp.tool()
async def hover(ref: str) -> str:
    """Hover over an element by snapshot ref."""
    try:
        loc = await SESSION.locate(ref)
        await loc.scroll_into_view_if_needed()
        await loc.hover()
        await _settle(SESSION.page, 100)
        return ok("Hover completed.", ref=ref)
    except Exception as exc:
        return fail("Hover failed.", ref=ref, detail=str(exc))


@mcp.tool()
async def scroll(
    direction: Literal["up", "down", "left", "right"] = "down",
    amount: int = 800,
    ref: Optional[str] = None,
) -> str:
    """
    Scroll the page, or bring a referenced element into view.

    For ref, no wheel movement is performed; the target is scrolled into view.
    """
    try:
        amount = max(1, min(int(amount), 20_000))

        if ref:
            loc = await SESSION.locate(ref)
            await loc.scroll_into_view_if_needed()
            return ok("Element scrolled into view.", ref=ref)

        dx = amount if direction == "right" else -amount if direction == "left" else 0
        dy = amount if direction == "down" else -amount if direction == "up" else 0

        await SESSION.page.mouse.wheel(dx, dy)
        await _settle(SESSION.page, 150)

        return ok(
            "Page scrolled.",
            direction=direction,
            amount=amount,
        )
    except Exception as exc:
        return fail("Scroll failed.", detail=str(exc))


@mcp.tool()
async def drag(from_ref: str, to_ref: str) -> str:
    """Drag one referenced element onto another."""
    try:
        src = await SESSION.locate(from_ref)
        dst = await SESSION.locate(to_ref)

        await src.scroll_into_view_if_needed()
        await dst.scroll_into_view_if_needed()
        await src.drag_to(dst)

        await _settle(SESSION.page)

        return ok(
            "Drag completed.",
            from_ref=from_ref,
            to_ref=to_ref,
        )
    except Exception as exc:
        return fail(
            "Drag failed.",
            from_ref=from_ref,
            to_ref=to_ref,
            detail=str(exc),
        )


@mcp.tool()
async def set_zoom(factor: float = 1.0) -> str:
    """Set CSS zoom on the page, e.g. 1.5 for 150%."""
    try:
        if not 0.25 <= factor <= 5:
            return fail("factor must be between 0.25 and 5.0.")

        await SESSION.page.evaluate(
            "(factor) => { document.documentElement.style.zoom = factor; }",
            factor,
        )

        return ok("Page zoom changed.", factor=factor)
    except Exception as exc:
        return fail("Zoom failed.", detail=str(exc))


@mcp.tool()
async def screenshot(
    full_page: bool = False,
    path: Optional[str] = None,
) -> Image | str:
    """
    Take a PNG screenshot.

    If path is omitted, returns the screenshot as an MCP Image.
    If path is provided, saves it to that local path and returns JSON metadata.
    """
    try:
        if path:
            destination = Path(path).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)

            await SESSION.page.screenshot(
                path=str(destination),
                full_page=full_page,
                type="png",
            )

            return ok(
                "Screenshot saved.",
                path=str(destination),
                full_page=full_page,
            )

        data = await SESSION.page.screenshot(
            full_page=full_page,
            type="png",
        )
        return Image(data=data, format="png")

    except Exception as exc:
        return fail("Screenshot failed.", detail=str(exc))


@mcp.tool()
async def copy_to_clipboard(text: str) -> str:
    """Write text to the browser/system clipboard."""
    try:
        await SESSION.page.evaluate(
            "(text) => navigator.clipboard.writeText(text)",
            text,
        )
        return ok("Copied text to clipboard.", characters=len(text))
    except Exception as exc:
        return fail(
            "Clipboard write failed.",
            detail=str(exc),
            recovery=(
                "The site/context may not permit clipboard access. "
                "Try a normal field fill or paste workflow instead."
            ),
        )


@mcp.tool()
async def paste_into(ref: str) -> str:
    """Focus a field and paste from the browser clipboard."""
    try:
        loc = await SESSION.locate(ref)
        await loc.scroll_into_view_if_needed()
        await loc.click()

        modifier = "Meta" if platform.system() == "Darwin" else "Control"
        await SESSION.page.keyboard.press(f"{modifier}+V")

        return ok("Clipboard paste completed.", ref=ref)
    except Exception as exc:
        return fail("Clipboard paste failed.", ref=ref, detail=str(exc))


@mcp.tool()
async def read_clipboard() -> str:
    """Read text from the browser clipboard."""
    try:
        text = await SESSION.page.evaluate(
            "() => navigator.clipboard.readText()"
        )
        return ok("Clipboard read completed.", text=text)
    except Exception as exc:
        return fail("Clipboard read failed.", detail=str(exc))


@mcp.tool()
async def run_javascript(code: str, args: Optional[list] = None) -> str:
    """
    Execute JavaScript in the active page and return a SMALL structured result.

    This tool is intentionally for structured JavaScript queries, not for
    dumping arbitrary page text. Prefer expressions such as:

        ({
            title: document.title,
            url: location.href,
            hasGame: typeof Game !== "undefined"
        })

    or a function:

        () => ({
            title: document.title,
            links: document.querySelectorAll("a").length
        })

    Bare object expressions are supported and are evaluated as expressions,
    so `{title: document.title}` does not accidentally become a JavaScript
    statement block.

    For large or intentionally textual data, use evaluate_text() instead.
    That tool applies an explicit character limit before the result enters the
    MCP conversation context.

    `args` is passed to the JavaScript function as ONE argument, matching
    Playwright's page.evaluate() API. For example:
        code="(args) => args.a + args.b", args={"a": 1, "b": 2}

    For backward compatibility, a list is also accepted and is exposed as
    the single `args` value.

    Errors are returned as structured JSON.
    """
    await SESSION.ensure_started()
    stripped = code.strip()

    # Detect actual function expressions.
    is_function = (
        stripped.startswith("function")
        or stripped.startswith("async function")
        or stripped.startswith("(") and "=>" in stripped
        or stripped.startswith("async ") and "=>" in stripped
        or re.match(r"^[A-Za-z_$][\w$]*\s*=>", stripped) is not None
    )

    if is_function:
        expression = stripped
    else:
        # Evaluate as an expression first. This makes object literals work:
        # ({title: document.title}) rather than treating them as a statement
        # block with labels.
        expression = f"() => ({stripped})"

    try:
        if args is not None:
            result = await SESSION.page.evaluate(expression, args)
        else:
            result = await SESSION.page.evaluate(expression)

        return json.dumps(
            {
                "ok": True,
                "result": result,
                "result_type": type(result).__name__,
            },
            ensure_ascii=False,
            default=str,
        )
    except Exception as e:
        return json.dumps(
            {
                "ok": False,
                "error": str(e),
                "hint": (
                    "Use evaluate_text() for intentionally retrieving page "
                    "text, and ensure structured JavaScript returns a "
                    "JSON-serializable value."
                ),
            },
            ensure_ascii=False,
        )


@mcp.tool()
async def evaluate_text(
    expression: str,
    max_chars: int = DEFAULT_EVALUATE_TEXT_MAX_CHARS,
) -> str:
    """
    Explicitly retrieve textual data from the active page with a hard size cap.

    This is deliberately separate from run_javascript(). Use it for things
    like:

        expression="document.body.innerText"
        max_chars=10000

    The expression may be a normal JavaScript expression or a function
    returning text. The server converts the result to text, normalizes it,
    and truncates it BEFORE returning it to the MCP client.

    `max_chars` defaults to 10,000 and is hard-capped by
    BROWSER_MCP_MAX_EVALUATE_TEXT_MAX_CHARS (default 100,000).

    The response includes metadata showing whether truncation occurred.
    """
    await SESSION.ensure_started()

    try:
        requested = int(max_chars)
    except (TypeError, ValueError):
        requested = DEFAULT_EVALUATE_TEXT_MAX_CHARS

    requested = max(1, min(requested, MAX_EVALUATE_TEXT_MAX_CHARS))

    stripped = expression.strip()
    if not stripped:
        return json.dumps(
            {"ok": False, "error": "expression cannot be empty."},
            ensure_ascii=False,
        )

    is_function = (
        stripped.startswith("function")
        or stripped.startswith("async function")
        or ("=>" in stripped and (
            stripped.startswith("(")
            or re.match(r"^[A-Za-z_$][\w$]*\s*=>", stripped) is not None
            or stripped.startswith("async ")
        ))
    )

    if is_function:
        wrapped = stripped
    else:
        wrapped = f"() => ({stripped})"

    try:
        result = await SESSION.page.evaluate(wrapped)

        if result is None:
            value = ""
        elif isinstance(result, str):
            value = result
        else:
            # Text mode is intentionally textual. JSON-encode structured
            # results rather than accidentally returning Python repr output.
            value = json.dumps(result, ensure_ascii=False, default=str)

        # Normalize excessive whitespace only at the edges. Do not collapse
        # internal newlines because this tool is specifically for page text.
        value = value.strip()

        original_chars = len(value)
        truncated = original_chars > requested
        output = value[:requested]

        payload = {
            "ok": True,
            "text": output,
            "chars": len(output),
            "original_chars": original_chars,
            "max_chars": requested,
            "truncated": truncated,
        }

        if truncated:
            payload["hint"] = (
                "Result was truncated before entering context. Increase "
                "max_chars only when needed."
            )

        return json.dumps(payload, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {
                "ok": False,
                "error": str(e),
                "max_chars": requested,
            },
            ensure_ascii=False,
        )


@mcp.tool()
async def get_console_log(
    clear: bool = False,
    limit: int = 100,
    types: Optional[list[str]] = None,
) -> str:
    """Return recent console messages and uncaught page errors."""
    try:
        limit = max(1, min(int(limit), MAX_CONSOLE_LOG))
        page_key = id(SESSION.page)
        log_items = SESSION.console_log(page_key)

        if types:
            wanted = {str(t).lower() for t in types}
            log_items = [
                item for item in log_items
                if str(item.get("type", "")).lower() in wanted
            ]

        trimmed = log_items[-limit:]

        if clear:
            SESSION._console_log[page_key] = []

        return ok(
            "Console log retrieved.",
            count=len(trimmed),
            cleared=clear,
            entries=trimmed,
        )
    except Exception as exc:
        return fail("Could not read console log.", detail=str(exc))


@mcp.tool()
async def list_tabs() -> str:
    """List open tabs with indexes, URLs, titles, and active state."""
    try:
        SESSION._sync_pages()

        tabs = []
        for i, page in enumerate(SESSION.pages):
            try:
                title = await page.title()
            except Exception:
                title = ""

            tabs.append(
                {
                    "index": i,
                    "active": i == SESSION.active_index,
                    "url": page.url,
                    "title": title,
                    "closed": page.is_closed(),
                }
            )

        return ok(
            "Tabs listed.",
            count=len(tabs),
            tabs=tabs,
        )
    except Exception as exc:
        return fail("Could not list tabs.", detail=str(exc))


@mcp.tool()
async def new_tab(url: Optional[str] = None) -> str:
    """Open a new tab, optionally navigate it, and make it active."""
    try:
        await SESSION.ensure_started()

        if SESSION.context is None:
            return fail("Browser context is unavailable.")

        page = await SESSION.context.new_page()
        SESSION._attach_page(page)

        SESSION.pages.append(page) if page not in SESSION.pages else None
        SESSION.active_index = SESSION.pages.index(page)

        if url:
            normalized = normalize_url(url)
            await _safe_goto(page, normalized)

        return ok(
            "New tab opened.",
            index=SESSION.active_index,
            url=page.url,
            title=await page.title(),
        )
    except Exception as exc:
        return fail("Could not open new tab.", detail=str(exc))


@mcp.tool()
async def switch_tab(index: int) -> str:
    """Switch the active tab by index."""
    try:
        SESSION._sync_pages()

        if not 0 <= index < len(SESSION.pages):
            return fail(
                "Invalid tab index.",
                requested=index,
                available=len(SESSION.pages),
            )

        SESSION.active_index = index
        page = SESSION.page
        await page.bring_to_front()

        return ok(
            "Active tab changed.",
            index=index,
            url=page.url,
            title=await page.title(),
        )
    except Exception as exc:
        return fail("Could not switch tabs.", detail=str(exc))


@mcp.tool()
async def close_tab(index: Optional[int] = None) -> str:
    """Close a tab, while keeping at least one tab open."""
    try:
        SESSION._sync_pages()

        if not SESSION.pages:
            return fail("No tabs are open.")

        idx = SESSION.active_index if index is None else index

        if not 0 <= idx < len(SESSION.pages):
            return fail("Invalid tab index.", requested=idx)

        if len(SESSION.pages) == 1:
            return fail("Refusing to close the last remaining tab.")

        page = SESSION.pages[idx]
        await page.close()

        # _drop_page_cache is called by the close listener, but synchronize
        # once more because event delivery can be asynchronous.
        SESSION._sync_pages()

        if SESSION.pages:
            SESSION.active_index = min(
                SESSION.active_index,
                len(SESSION.pages) - 1,
            )

        return ok(
            "Tab closed.",
            closed_index=idx,
            remaining_tabs=len(SESSION.pages),
            active_tab=SESSION.active_index if SESSION.pages else None,
        )
    except Exception as exc:
        return fail("Could not close tab.", detail=str(exc))


@mcp.tool()
async def go_back() -> str:
    """Navigate back in the active tab."""
    try:
        page = SESSION.page
        response = await page.go_back(
            wait_until="domcontentloaded",
            timeout=ACTION_TIMEOUT_MS,
        )
        await _settle(page)

        return ok(
            "Went back.",
            url=page.url,
            title=await page.title(),
            had_response=response is not None,
        )
    except Exception as exc:
        return fail("Back navigation failed.", detail=str(exc))


@mcp.tool()
async def go_forward() -> str:
    """Navigate forward in the active tab."""
    try:
        page = SESSION.page
        response = await page.go_forward(
            wait_until="domcontentloaded",
            timeout=ACTION_TIMEOUT_MS,
        )
        await _settle(page)

        return ok(
            "Went forward.",
            url=page.url,
            title=await page.title(),
            had_response=response is not None,
        )
    except Exception as exc:
        return fail("Forward navigation failed.", detail=str(exc))


@mcp.tool()
async def reload_page() -> str:
    """Reload the active page."""
    try:
        page = SESSION.page
        await page.reload(
            wait_until="domcontentloaded",
            timeout=ACTION_TIMEOUT_MS,
        )
        await _settle(page)

        return ok(
            "Page reloaded.",
            url=page.url,
            title=await page.title(),
        )
    except Exception as exc:
        return fail("Reload failed.", detail=str(exc))


@mcp.tool()
async def get_page_info() -> str:
    """Get lightweight active-page information without a full snapshot."""
    try:
        page = SESSION.page
        info = await page.evaluate(
            """() => ({
                x: window.scrollX,
                y: window.scrollY,
                height: document.documentElement.scrollHeight,
                width: document.documentElement.scrollWidth,
                viewport_width: window.innerWidth,
                viewport_height: window.innerHeight
            })"""
        )

        return ok(
            "Page information retrieved.",
            url=page.url,
            title=await page.title(),
            scroll=info,
            tab_count=len(SESSION.pages),
            active_tab=SESSION.active_index,
        )
    except Exception as exc:
        return fail("Could not get page info.", detail=str(exc))


@mcp.tool()
async def wait(
    seconds: float = 1.0,
    until: Optional[str] = None,
) -> str:
    """
    Wait for a short period.

    If until is provided, wait for a CSS selector to become visible.
    """
    try:
        seconds = max(0.0, min(float(seconds), 30.0))

        if until:
            timeout = int(max(1.0, seconds) * 1000)
            await SESSION.page.locator(until).first.wait_for(
                state="visible",
                timeout=timeout,
            )
            return ok(
                "Selector became visible.",
                selector=until,
                waited_seconds=seconds,
            )

        await asyncio.sleep(seconds)
        return ok("Wait completed.", seconds=seconds)

    except PWTimeoutError:
        return fail(
            "Timed out waiting for selector.",
            selector=until,
            seconds=seconds,
        )
    except Exception as exc:
        return fail("Wait failed.", detail=str(exc))


@mcp.tool()
async def stop_browser() -> str:
    """Close Chromium and end the persistent browser session."""
    try:
        await SESSION.close()
        return ok("Browser closed.")
    except Exception as exc:
        return fail("Browser shutdown failed.", detail=str(exc))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Playwright browser-control MCP server."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium headlessly instead of visibly.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Override BROWSER_MCP_PROFILE_DIR.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    global PROFILE_DIR
    if args.profile:
        PROFILE_DIR = Path(args.profile).expanduser().resolve()

    if args.headless:
        os.environ["BROWSER_MCP_HEADLESS"] = "1"

    log(
        f"Starting browser-mcp with MCP SDK v{MCP_SDK_MAJOR}; "
        f"profile={PROFILE_DIR}"
    )

    # stdio is deliberately the default: MCP hosts communicate over stdout,
    # so never put normal diagnostic print() calls there.
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
