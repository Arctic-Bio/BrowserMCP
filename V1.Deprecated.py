"""
browser_mcp.py — a self-contained local MCP server for AI-driven browser control.

WHAT THIS IS
    A single-file Model Context Protocol (MCP) server, built on Playwright,
    that gives an AI agent tools to browse, click, type, scroll, drag,
    switch tabs, screenshot, copy/paste, and — the important part — read a
    page through a custom, context-aware element tree instead of raw HTML.

    Instead of exposing generic ARIA roles, every meaningful element on a
    page is classified into one of a small set of *intent* tags:

        action     - clickable things that DO something (buttons, toggles,
                      checkboxes, elements with click handlers)
        link       - navigational elements (<a href=...>)
        input      - text fields, textareas, selects, file uploads
        title      - headings / section titles
        visual     - images, icons, video, canvas
        container  - structural landmarks (nav, header, footer, form, dialog)
        text       - meaningful body copy / labels

    Each element gets a stable reference id (e1, e2, ...) written onto the
    DOM as a data attribute, so the agent can act on it in later tool calls
    without re-locating it. Snapshots are diffed against the previous one
    so repeat calls only describe what changed, which keeps token usage low
    and responses fast.

INSTALL
    pip install mcp playwright --break-system-packages     # or in a venv, drop the flag
    python3 -m playwright install chromium

    (This file auto-detects both mcp SDK v1 (`FastMCP`) and v2 (`MCPServer`).)

RUN STANDALONE (for testing)
    python3 browser_mcp.py

REGISTER WITH AN MCP CLIENT (example: Claude Desktop / Claude Code config)
    {
      "mcpServers": {
        "browser": {
          "command": "python3",
          "args": ["/absolute/path/to/browser_mcp.py"]
        }
      }
    }

NOTES ON THE BROWSER PROFILE
    This launches a *persistent* Chromium profile stored under
    ~/.browser_mcp_profile (configurable via BROWSER_MCP_PROFILE_DIR env
    var). It is not your everyday Chrome profile (Chrome won't let two
    processes share a profile, and modern Chrome blocks automation flags
    on your default profile) — it's a dedicated automation profile. The
    first time you log into a site through it, that session persists for
    next time, same as a normal browser.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import hashlib
import platform
from pathlib import Path
from typing import Any, Literal, Optional

# --------------------------------------------------------------------------
# MCP SDK import shim (supports both v1 `FastMCP` and v2 `MCPServer` names)
# --------------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP  # mcp < 2.0
    try:
        from mcp.server.fastmcp import Image
    except ImportError:
        from mcp.server.fastmcp.utilities.types import Image
except ImportError:
    from mcp.server.mcpserver import MCPServer as FastMCP  # mcp >= 2.0
    from mcp.server.mcpserver.utilities.types import Image

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
    Locator,
    TimeoutError as PWTimeoutError,
)

# ==========================================================================
# Configuration
# ==========================================================================

PROFILE_DIR = Path(os.environ.get("BROWSER_MCP_PROFILE_DIR", str(Path.home() / ".browser_mcp_profile")))
HEADLESS_DEFAULT = os.environ.get("BROWSER_MCP_HEADLESS", "0") == "1"
DEFAULT_VIEWPORT = {"width": 1440, "height": 900}
MAX_TEXT_NODES = 150          # cap on 'text' category elements per snapshot, to control noise
LABEL_MAX_LEN = 200
NAV_SETTLE_MS = 300           # short settle wait after navigation/clicks before snapshotting
MAX_CONSOLE_LOG = 500         # cap on buffered console messages per page

# ==========================================================================
# The injected extraction + classification script
# ==========================================================================
# This runs inside the page. It:
#   1. Assigns each candidate element a stable ref id (persisted on
#      `window.__mcpState` so it survives repeated evaluate() calls on the
#      same page, and written to a `data-mcp-ref` attribute so Python can
#      re-locate the element by a plain CSS selector).
#   2. Classifies each element into the intent taxonomy described above.
#   3. Computes a best-effort accessible label.
#   4. Walks up the tree to find the nearest semantic "region" landmark.
#   5. Returns a flat JSON array (not a nested tree) — flat is denser,
#      easier for the agent to filter/search, and trivial to diff.

EXTRACTION_JS = r"""
() => {
    if (!window.__mcpState) {
        window.__mcpState = { counter: 0, refMap: new WeakMap() };
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
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
        if (parseFloat(style.opacity) === 0) return false;
        const rect = el.getBoundingClientRect();
        if (rect.width <= 0 && rect.height <= 0) return false;
        if (el.hasAttribute('aria-hidden') && el.getAttribute('aria-hidden') === 'true') return false;
        if (el.hidden) return false;
        return true;
    }

    function accessibleName(el) {
        const candidates = [
            el.getAttribute('aria-label'),
            el.getAttribute('alt'),
            el.getAttribute('placeholder'),
            el.getAttribute('title'),
            el.tagName.toLowerCase() === 'input' ? el.value : null,
            el.innerText,
        ];
        for (const c of candidates) {
            if (c && c.trim().length > 0) {
                return c.trim().replace(/\s+/g, ' ').slice(0, 200);
            }
        }
        return '';
    }

    const LANDMARK_SELECTOR = [
        'nav', 'header', 'footer', 'main', 'form', 'dialog', 'aside',
        '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]',
        '[role="main"]', '[role="dialog"]', '[role="form"]', 'section[aria-label]'
    ].join(',');

    function nearestRegion(el) {
        let cur = el.parentElement;
        let hops = 0;
        while (cur && hops < 40) {
            if (cur.matches && cur.matches(LANDMARK_SELECTOR)) {
                const label = cur.getAttribute('aria-label');
                return label ? label.slice(0, 60) : cur.tagName.toLowerCase();
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
        const hasHandler = !!(el.onclick || el.getAttribute('onclick'));

        if (['textarea', 'select'].includes(tag)) return 'input';
        if (tag === 'input' && !['button', 'submit', 'checkbox', 'radio', 'image', 'reset'].includes(type)) return 'input';
        if (tag === 'input' && ['checkbox', 'radio'].includes(type)) return 'action';
        if (tag === 'button' || role === 'button' || type === 'submit' || type === 'button' || hasHandler) return 'action';
        if (['checkbox', 'radio', 'switch', 'tab', 'menuitem', 'option'].includes(role)) return 'action';
        if (tag === 'a' && el.hasAttribute('href')) return 'link';
        if (/^h[1-6]$/.test(tag) || role === 'heading') return 'title';
        if (['img', 'svg', 'canvas', 'video', 'picture'].includes(tag)) return 'visual';
        const landmarkTags = ['nav', 'header', 'footer', 'main', 'form', 'dialog', 'aside'];
        if (landmarkTags.includes(tag) || ['navigation', 'banner', 'contentinfo', 'main', 'dialog', 'region'].includes(role)) return 'container';
        return 'text';
    }

    const SELECTOR = [
        'a[href]', 'button', '[role="button"]', '[onclick]',
        'input', 'textarea', 'select',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', '[role="heading"]',
        'img', 'svg[aria-label]', 'video', 'canvas',
        'nav', 'header', 'footer', 'main', 'form', 'dialog',
        '[role="navigation"]', '[role="banner"]', '[role="dialog"]',
        'p', 'li', 'label', 'td', 'th', 'span[class*="price"]',
        '[role="checkbox"]', '[role="radio"]', '[role="switch"]',
        '[role="tab"]', '[role="menuitem"]', '[role="option"]'
    ].join(',');

    const seen = new Set();
    const results = [];
    let textCount = 0;
    const vw = window.innerWidth, vh = window.innerHeight, sy = window.scrollY, sx = window.scrollX;

    const nodeList = document.querySelectorAll(SELECTOR);
    for (const el of nodeList) {
        if (seen.has(el)) continue;
        seen.add(el);
        if (!isVisible(el)) continue;

        const category = classify(el);

        // Cap noisy 'text' category so it doesn't drown out interactive elements.
        if (category === 'text') {
            const txt = (el.innerText || '').trim();
            if (txt.length < 2 || txt.length > 300) continue;
            // skip if this element's only content duplicates a single child we'll also see
            if (el.children.length === 1 && el.children[0].innerText === el.innerText) continue;
            textCount += 1;
            if (textCount > __MAX_TEXT_NODES__) continue;
        }

        const rect = el.getBoundingClientRect();
        const ref = getRef(el);

        // Containers/landmarks should NOT dump their full nested innerText
        // (that would include every descendant's text and blow up tokens) —
        // just use an aria-label or the first heading inside them, if any.
        let label;
        if (category === 'container') {
            label = el.getAttribute('aria-label') || '';
            if (!label) {
                const heading = el.querySelector('h1,h2,h3,h4,h5,h6,[role="heading"]');
                if (heading) label = (heading.innerText || '').trim().slice(0, 80);
            }
        } else {
            label = accessibleName(el) || (category === 'visual' ? '(unlabeled ' + el.tagName.toLowerCase() + ')' : '');
        }

        const entry = {
            ref: ref,
            tag: category,
            html_tag: el.tagName.toLowerCase(),
            label: label,
            region: nearestRegion(el),
            bbox: [Math.round(rect.left), Math.round(rect.top), Math.round(rect.width), Math.round(rect.height)],
            in_viewport: rect.bottom > 0 && rect.top < vh && rect.right > 0 && rect.left < vw,
        };

        if (category === 'link') entry.href = el.getAttribute('href');
        if (category === 'title') {
            const m = el.tagName.toLowerCase().match(/^h([1-6])$/);
            entry.level = m ? parseInt(m[1]) : (el.getAttribute('aria-level') || null);
        }
        if (category === 'input') {
            entry.input_type = el.tagName.toLowerCase() === 'input' ? (el.getAttribute('type') || 'text') : el.tagName.toLowerCase();
            entry.value = (el.value || '').slice(0, 200);
        }
        if (category === 'action') {
            entry.disabled = !!el.disabled;
            if (['checkbox', 'radio'].includes((el.getAttribute('type') || '').toLowerCase()) || ['checkbox', 'radio', 'switch'].includes(role_of(el))) {
                entry.checked = !!el.checked || el.getAttribute('aria-checked') === 'true';
            }
        }

        results.push(entry);
    }

    function role_of(el) { return (el.getAttribute('role') || '').toLowerCase(); }

    return {
        url: window.location.href,
        title: document.title,
        scroll: { x: sx, y: sy, viewport_width: vw, viewport_height: vh,
                  page_height: document.documentElement.scrollHeight },
        elements: results,
    };
}
""".replace("__MAX_TEXT_NODES__", str(MAX_TEXT_NODES))


# ==========================================================================
# Browser session manager
# ==========================================================================

class BrowserSession:
    """Owns the Playwright lifecycle, tabs, and per-page snapshot cache."""

    def __init__(self) -> None:
        self._pw: Optional[Playwright] = None
        self.context: Optional[BrowserContext] = None
        self.pages: list[Page] = []
        self.active_index: int = 0
        # ref -> element dict, per page (keyed by page's id())
        self._last_snapshot: dict[int, dict[str, dict]] = {}
        # console message ring-buffer per page (keyed by page's id())
        self._console_log: dict[int, list[dict]] = {}

    async def ensure_started(self, headless: bool = HEADLESS_DEFAULT) -> None:
        if self.context is not None:
            return
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        self._pw = await async_playwright().start()
        self.context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=headless,
            viewport=DEFAULT_VIEWPORT,
            args=["--disable-blink-features=AutomationControlled"],
        )
        try:
            await self.context.grant_permissions(["clipboard-read", "clipboard-write"])
        except Exception:
            pass  # not fatal — clipboard tools will fall back gracefully

        if self.context.pages:
            self.pages = list(self.context.pages)
        else:
            self.pages = [await self.context.new_page()]
        self.active_index = 0
        self.context.on("page", self._on_new_page)
        for p in self.pages:
            self._attach_console_listener(p)

    def _on_new_page(self, page: Page) -> None:
        if page not in self.pages:
            self.pages.append(page)
        self._attach_console_listener(page)

    def _attach_console_listener(self, page: Page) -> None:
        key = id(page)
        if key in self._console_log:
            return
        self._console_log[key] = []

        def _on_console(msg) -> None:
            buf = self._console_log.setdefault(key, [])
            buf.append({
                "type": msg.type,
                "text": msg.text,
                "time": time.time(),
            })
            if len(buf) > MAX_CONSOLE_LOG:
                del buf[: len(buf) - MAX_CONSOLE_LOG]

        def _on_pageerror(exc) -> None:
            buf = self._console_log.setdefault(key, [])
            buf.append({
                "type": "pageerror",
                "text": str(exc),
                "time": time.time(),
            })
            if len(buf) > MAX_CONSOLE_LOG:
                del buf[: len(buf) - MAX_CONSOLE_LOG]

        page.on("console", _on_console)
        page.on("pageerror", _on_pageerror)

    @property
    def page(self) -> Page:
        if not self.pages:
            raise RuntimeError("No browser pages open. Call navigate() first.")
        self.active_index = min(self.active_index, len(self.pages) - 1)
        return self.pages[self.active_index]

    async def close(self) -> None:
        if self.context:
            await self.context.close()
        if self._pw:
            await self._pw.stop()
        self.context = None

    async def locate(self, ref: str) -> Locator:
        loc = self.page.locator(f'[data-mcp-ref="{ref}"]').first
        count = await loc.count()
        if count == 0:
            raise ValueError(
                f"No element found for ref '{ref}'. It may be stale — take a new snapshot() first."
            )
        return loc

    async def extract(self) -> dict:
        await self.page.wait_for_timeout(NAV_SETTLE_MS)
        raw = await self.page.evaluate(EXTRACTION_JS)
        return raw

    def diff_against_last(self, page_key: int, elements: list[dict]) -> dict:
        current = {el["ref"]: el for el in elements}
        previous = self._last_snapshot.get(page_key, {})

        added, changed, removed = [], [], []
        for ref, el in current.items():
            if ref not in previous:
                added.append(el)
            elif previous[ref] != el:
                changed.append(el)
        for ref in previous:
            if ref not in current:
                removed.append(ref)

        self._last_snapshot[page_key] = current
        return {"added": added, "changed": changed, "removed_refs": removed,
                "total_elements": len(current)}

    def full_snapshot_cache(self, page_key: int) -> dict[str, dict]:
        return self._last_snapshot.get(page_key, {})

    def console_log(self, page_key: int) -> list[dict]:
        return self._console_log.get(page_key, [])


SESSION = BrowserSession()

# ==========================================================================
# MCP server + tools
# ==========================================================================

mcp = FastMCP("browser-mcp")


def _fmt_elements(elements: list[dict], limit: Optional[int] = None) -> list[dict]:
    """Trim element dicts down to what the agent actually needs, to save tokens."""
    out = []
    for el in elements[: limit] if limit else elements:
        item = {"ref": el["ref"], "tag": el["tag"], "label": el["label"], "region": el["region"]}
        for extra in ("href", "level", "input_type", "value", "disabled", "checked", "in_viewport"):
            if extra in el:
                item[extra] = el[extra]
        out.append(item)
    return out


@mcp.tool()
async def start_browser(headless: bool = False) -> str:
    """Start the persistent Chromium session. Safe to call multiple times (no-op if already running)."""
    await SESSION.ensure_started(headless=headless)
    return f"Browser started. Profile: {PROFILE_DIR}. Tabs open: {len(SESSION.pages)}."


@mcp.tool()
async def navigate(url: str) -> str:
    """Navigate the active tab to a URL. Adds https:// automatically if no scheme is given."""
    await SESSION.ensure_started()
    if not url.startswith(("http://", "https://", "file://", "about:")):
        url = "https://" + url
    await SESSION.page.goto(url, wait_until="domcontentloaded")
    return f"Navigated to {SESSION.page.url}"


@mcp.tool()
async def snapshot(mode: Literal["full", "diff"] = "diff", viewport_only: bool = False) -> str:
    """
    Get a context-aware, classified snapshot of the current page's interactive
    and meaningful elements (buttons, links, inputs, titles, images, text).

    mode='diff' (default) returns only what's new/changed/removed since the
    last snapshot of this page — much cheaper on tokens for repeated calls.
    mode='full' returns every classified element.

    Each element has a stable 'ref' id (e.g. "e14") — pass that ref into
    click(), type_text(), drag(), etc. to act on it.
    """
    await SESSION.ensure_started()
    raw = await SESSION.extract()
    elements = raw["elements"]
    if viewport_only:
        elements = [e for e in elements if e.get("in_viewport")]

    page_key = id(SESSION.page)
    if mode == "full":
        SESSION._last_snapshot[page_key] = {e["ref"]: e for e in elements}
        payload = {
            "url": raw["url"], "title": raw["title"], "scroll": raw["scroll"],
            "elements": _fmt_elements(elements),
        }
    else:
        d = SESSION.diff_against_last(page_key, elements)
        payload = {
            "url": raw["url"], "title": raw["title"], "scroll": raw["scroll"],
            "added": _fmt_elements(d["added"]),
            "changed": _fmt_elements(d["changed"]),
            "removed_refs": d["removed_refs"],
            "total_elements": d["total_elements"],
        }
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
async def search_elements(tag: Optional[str] = None, contains: Optional[str] = None) -> str:
    """
    Search the MOST RECENT snapshot's cached elements without touching the
    browser — fast, zero-latency filtering. tag: one of action/link/input/
    title/visual/container/text. contains: case-insensitive substring match
    against the element's label.
    """
    page_key = id(SESSION.page)
    cache = SESSION.full_snapshot_cache(page_key)
    if not cache:
        return json.dumps({"error": "No snapshot cached yet — call snapshot() first."})
    results = list(cache.values())
    if tag:
        results = [e for e in results if e["tag"] == tag]
    if contains:
        needle = contains.lower()
        results = [e for e in results if needle in e.get("label", "").lower()]
    return json.dumps(_fmt_elements(results), ensure_ascii=False)


@mcp.tool()
async def click(ref: str, button: Literal["left", "right", "middle"] = "left", double: bool = False) -> str:
    """Click an element by its snapshot ref id."""
    loc = await SESSION.locate(ref)
    await loc.scroll_into_view_if_needed()
    if double:
        await loc.dblclick(button=button)
    else:
        await loc.click(button=button)
    await SESSION.page.wait_for_timeout(NAV_SETTLE_MS)
    return f"Clicked {ref}."


@mcp.tool()
async def multi_click(
    ref: Optional[str] = None,
    count: int = 2,
    delay_ms: int = 250,
    button: Literal["left", "right", "middle"] = "left",
    sequence: Optional[list[dict]] = None,
) -> str:
    """
    Click one or more elements over time, waiting between clicks. Useful for
    rapid-fire click targets (games, counters, "load more" buttons that need
    several hits), or for a timed sequence of clicks across different
    elements (e.g. open a menu, wait, then click an item that only appears
    after the menu animates in).

    Two modes:
      - Single-target repeat: pass `ref`, `count` (how many clicks), and
        `delay_ms` (pause between each click). Re-locates the element by ref
        before every click, so it still works if the DOM re-renders between
        clicks (as long as the ref is still attached).
      - Sequence: pass `sequence`, a list of steps, each a dict with keys
        `ref` (required), `delay_ms` (pause *before* this click, default 0),
        and optionally `button` ("left"/"right"/"middle", default "left").
        Steps run in order. Use this to click different elements with
        different timing in one call instead of many round trips.

    If both `ref` and `sequence` are omitted, an error is returned.
    """
    results: list[str] = []

    if sequence:
        for i, step in enumerate(sequence):
            step_ref = step.get("ref")
            if not step_ref:
                results.append(f"[step {i}] skipped: missing 'ref'.")
                continue
            wait_ms = int(step.get("delay_ms", 0) or 0)
            step_button = step.get("button", "left")
            if wait_ms > 0:
                await asyncio.sleep(wait_ms / 1000)
            try:
                loc = await SESSION.locate(step_ref)
                await loc.scroll_into_view_if_needed()
                await loc.click(button=step_button)
                results.append(f"[step {i}] clicked {step_ref} (waited {wait_ms}ms).")
            except Exception as e:
                results.append(f"[step {i}] failed to click {step_ref}: {e}")
        await SESSION.page.wait_for_timeout(NAV_SETTLE_MS)
        return "\n".join(results)

    if not ref:
        return "Error: provide either 'ref' (for a repeated single-target click) or 'sequence'."

    count = max(1, count)
    for i in range(count):
        try:
            loc = await SESSION.locate(ref)
            await loc.scroll_into_view_if_needed()
            await loc.click(button=button)
            results.append(f"[click {i + 1}/{count}] clicked {ref}.")
        except Exception as e:
            results.append(f"[click {i + 1}/{count}] failed: {e}")
            break
        if i < count - 1 and delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)
    await SESSION.page.wait_for_timeout(NAV_SETTLE_MS)
    return "\n".join(results)


@mcp.tool()
async def type_text(ref: str, text: str, clear_first: bool = True, press_enter: bool = False) -> str:
    """Type text into an input/textarea element by its snapshot ref id."""
    loc = await SESSION.locate(ref)
    await loc.scroll_into_view_if_needed()
    await loc.click()
    if clear_first:
        await loc.fill("")
    await loc.type(text, delay=15)
    if press_enter:
        await loc.press("Enter")
    return f"Typed into {ref}."


@mcp.tool()
async def press_key(key: str) -> str:
    """Press a keyboard key at the page level (e.g. 'Escape', 'Tab', 'ArrowDown', 'Control+A')."""
    await SESSION.page.keyboard.press(key)
    return f"Pressed {key}."


@mcp.tool()
async def scroll(direction: Literal["up", "down", "left", "right"] = "down", amount: int = 800, ref: Optional[str] = None) -> str:
    """Scroll the page in a direction, or scroll a specific element (by ref) into view."""
    if ref:
        loc = await SESSION.locate(ref)
        await loc.scroll_into_view_if_needed()
        return f"Scrolled {ref} into view."
    dx = amount if direction == "right" else -amount if direction == "left" else 0
    dy = amount if direction == "down" else -amount if direction == "up" else 0
    await SESSION.page.mouse.wheel(dx, dy)
    await SESSION.page.wait_for_timeout(150)
    return f"Scrolled {direction} by {amount}px."


@mcp.tool()
async def drag(from_ref: str, to_ref: str) -> str:
    """Drag an element (from_ref) and drop it onto another element (to_ref)."""
    src = await SESSION.locate(from_ref)
    dst = await SESSION.locate(to_ref)
    await src.scroll_into_view_if_needed()
    await src.drag_to(dst)
    return f"Dragged {from_ref} -> {to_ref}."


@mcp.tool()
async def set_zoom(factor: float = 1.0) -> str:
    """Set page CSS zoom (e.g. 1.5 for 150%). Useful for 'enlarging' hard-to-read pages."""
    await SESSION.page.evaluate(f"document.body.style.zoom = {factor};")
    return f"Zoom set to {factor}."


@mcp.tool()
async def screenshot(full_page: bool = False) -> Image:
    """Take a screenshot of the current tab (viewport by default, or the full scrollable page)."""
    data = await SESSION.page.screenshot(full_page=full_page, type="png")
    return Image(data=data, format="png")


@mcp.tool()
async def copy_to_clipboard(text: str) -> str:
    """Write text to the system/browser clipboard."""
    try:
        await SESSION.page.evaluate("(t) => navigator.clipboard.writeText(t)", text)
        return "Copied to clipboard."
    except Exception as e:
        return f"Clipboard write failed ({e}). The page may not allow clipboard access."


@mcp.tool()
async def paste_into(ref: str) -> str:
    """Focus an element by ref and simulate a real paste keystroke (Ctrl+V / Cmd+V) from the clipboard."""
    loc = await SESSION.locate(ref)
    await loc.click()
    mod = "Meta" if platform.system() == "Darwin" else "Control"
    await SESSION.page.keyboard.press(f"{mod}+V")
    return f"Pasted into {ref}."


@mcp.tool()
async def read_clipboard() -> str:
    """Read the current text clipboard contents."""
    try:
        text = await SESSION.page.evaluate("() => navigator.clipboard.readText()")
        return text
    except Exception as e:
        return f"Clipboard read failed ({e})."


@mcp.tool()
async def run_javascript(code: str, args: Optional[list] = None) -> str:
    """
    Execute JavaScript in the active page's context (like typing into DevTools'
    console) and return the result as JSON.

    `code` can be either a bare expression ("document.title") or a function
    body / full function. Playwright's evaluate() accepts either a plain
    expression or a function; if `code` doesn't look like a function
    already (doesn't start with 'function' or contain '=>'), it's wrapped
    as `() => { <code> }` automatically so multi-statement snippets work,
    e.g. code="const x = 1 + 1; return x;".

    `args` (optional list) is passed through and available as `arguments`
    style positional params to a function you supply directly in `code`
    (e.g. code="(a, b) => a + b", args=[1, 2]).

    The return value must be JSON-serializable (objects, arrays, strings,
    numbers, booleans, null) — DOM nodes, functions, etc. won't survive the
    round trip. Errors thrown in the page are caught and returned as an
    error string rather than raising.

    Also see get_console_log() to read console.log/warn/error output and
    uncaught page errors captured passively while the page runs, separate
    from whatever this call returns.
    """
    await SESSION.ensure_started()
    stripped = code.strip()
    is_function = stripped.startswith("function") or stripped.startswith("async function") or "=>" in stripped.split("\n", 1)[0]
    if not is_function:
        wrapped = "() => { " + code + " }"
    else:
        wrapped = code
    try:
        if args:
            result = await SESSION.page.evaluate(wrapped, args)
        else:
            result = await SESSION.page.evaluate(wrapped)
        return json.dumps({"result": result}, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def get_console_log(clear: bool = False, limit: int = 100) -> str:
    """
    Return recent browser console messages (console.log/info/warn/error and
    uncaught page errors) captured passively for the active tab since it was
    opened (or since the log was last cleared). Newest messages last.

    clear: if true, empties the buffer for this tab after reading it.
    limit: max number of most-recent messages to return (default 100).
    """
    page_key = id(SESSION.page)
    log = SESSION.console_log(page_key)
    trimmed = log[-limit:] if limit else list(log)
    if clear:
        SESSION._console_log[page_key] = []
    return json.dumps(trimmed, ensure_ascii=False)


@mcp.tool()
async def list_tabs() -> str:
    """List open tabs with their index, title, and URL. The active tab is marked."""
    out = []
    for i, p in enumerate(SESSION.pages):
        out.append({"index": i, "url": p.url, "title": await p.title(), "active": i == SESSION.active_index})
    return json.dumps(out, ensure_ascii=False)


@mcp.tool()
async def new_tab(url: Optional[str] = None) -> str:
    """Open a new tab, optionally navigating it to a URL, and make it active."""
    await SESSION.ensure_started()
    page = await SESSION.context.new_page()
    if page not in SESSION.pages:
        SESSION.pages.append(page)
    SESSION.active_index = SESSION.pages.index(page)
    if url:
        if not url.startswith(("http://", "https://", "file://", "about:")):
            url = "https://" + url
        await page.goto(url, wait_until="domcontentloaded")
    return f"Opened new tab at index {SESSION.active_index}."


@mcp.tool()
async def switch_tab(index: int) -> str:
    """Switch the active tab by index (see list_tabs())."""
    if not (0 <= index < len(SESSION.pages)):
        return f"Invalid tab index {index}. Only {len(SESSION.pages)} tab(s) open."
    SESSION.active_index = index
    await SESSION.page.bring_to_front()
    return f"Switched to tab {index}: {SESSION.page.url}"


@mcp.tool()
async def close_tab(index: Optional[int] = None) -> str:
    """Close a tab by index (defaults to the active tab). Keeps at least one tab open."""
    idx = SESSION.active_index if index is None else index
    if not (0 <= idx < len(SESSION.pages)):
        return f"Invalid tab index {idx}."
    if len(SESSION.pages) == 1:
        return "Refusing to close the last remaining tab."
    page = SESSION.pages.pop(idx)
    await page.close()
    SESSION.active_index = min(SESSION.active_index, len(SESSION.pages) - 1)
    return f"Closed tab {idx}."


@mcp.tool()
async def go_back() -> str:
    """Navigate back in the active tab's history."""
    await SESSION.page.go_back(wait_until="domcontentloaded")
    return f"Went back to {SESSION.page.url}"


@mcp.tool()
async def go_forward() -> str:
    """Navigate forward in the active tab's history."""
    await SESSION.page.go_forward(wait_until="domcontentloaded")
    return f"Went forward to {SESSION.page.url}"


@mcp.tool()
async def reload_page() -> str:
    """Reload the active tab."""
    await SESSION.page.reload(wait_until="domcontentloaded")
    return f"Reloaded {SESSION.page.url}"


@mcp.tool()
async def get_page_info() -> str:
    """Get the active tab's URL, title, and scroll position without a full snapshot."""
    p = SESSION.page
    scroll = await p.evaluate("() => ({x: window.scrollX, y: window.scrollY, height: document.documentElement.scrollHeight})")
    return json.dumps({"url": p.url, "title": await p.title(), "scroll": scroll, "tab_count": len(SESSION.pages)})


@mcp.tool()
async def wait(seconds: float = 1.0) -> str:
    """Pause for a number of seconds (e.g. to let an animation or async load finish)."""
    await asyncio.sleep(min(seconds, 30))
    return f"Waited {seconds}s."


@mcp.tool()
async def stop_browser() -> str:
    """Close the browser and end the session."""
    await SESSION.close()
    return "Browser closed."


# ==========================================================================
# Entrypoint
# ==========================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
