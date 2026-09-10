"""Scrape Sphinx AutoAPI docs from https://captainglac1er.github.io/rocket-league-gym/

Prefer `_sources/*.rst.txt` (clean RST) when available; fall back to HTML→markdown.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import html2text

BASE = "https://captainglac1er.github.io/rocket-league-gym/"
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "rlgym-api"
UA = "rl-bot-local-docs/0.1 (offline instructional mirror)"

h2t = html2text.HTML2Text()
h2t.body_width = 0
h2t.ignore_images = True
h2t.protect_links = True


def fetch(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def docnames() -> list[tuple[str, str]]:
    js = fetch(BASE + "searchindex.js").decode("utf-8", "replace")
    m = re.search(r"Search\.setIndex\((\{.*\})\)\s*$", js, re.S)
    if not m:
        raise RuntimeError("Could not parse searchindex.js")
    idx = json.loads(m.group(1))
    return list(zip(idx["docnames"], idx["titles"]))


def try_source(docname: str) -> str | None:
    url = f"{BASE}_sources/{docname}.rst.txt"
    try:
        return fetch(url).decode("utf-8", "replace")
    except Exception:
        return None


def html_to_markdown(docname: str) -> str:
    html = fetch(f"{BASE}{docname}.html")
    soup = BeautifulSoup(html, "lxml")
    body = soup.select_one("div[role='main']") or soup.select_one(".document")
    if body is None:
        raise RuntimeError(f"No main content for {docname}")
    for sel in ["nav", ".sphinxsidebar", ".related", ".headerlink"]:
        for el in body.select(sel):
            el.decompose()
    return h2t.handle(str(body)).strip()


def main() -> None:
    docs = docnames()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows: list[tuple[str, str]] = []

    for i, (docname, title) in enumerate(docs, 1):
        print(f"[{i}/{len(docs)}] {docname}")
        source = try_source(docname)
        rel = Path(docname).with_suffix(".md" if source is None else ".rst")
        dest = OUT_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        if source is not None:
            body = source
            fmt_note = "Sphinx RST source"
        else:
            body = html_to_markdown(docname) + "\n"
            fmt_note = "HTML→markdown fallback"

        url = f"{BASE}{docname}.html"
        header = (
            f".. scraped-from: {url}\n"
            f".. format: {fmt_note}\n\n"
            if dest.suffix == ".rst"
            else (
                f"# {title}\n\n"
                f"> Source: [{url}]({url})  \n"
                f"> Format: {fmt_note}\n\n"
            )
        )
        # For RST, keep as comment-ish preamble then raw source
        if dest.suffix == ".rst":
            text = (
                f".. Scraped for local reference from the public RLGym API docs.\n"
                f".. Source: {url}\n"
                f".. Format: {fmt_note}\n\n"
                f"{body}"
            )
            if not text.endswith("\n"):
                text += "\n"
        else:
            text = header + body
            if not text.endswith("\n"):
                text += "\n"

        dest.write_text(text, encoding="utf-8")
        index_rows.append((title, rel.as_posix()))
        time.sleep(0.2)

    lines = [
        "# RLGym API reference (local mirror)",
        "",
        "AutoAPI pages from [captainglac1er.github.io/rocket-league-gym](https://captainglac1er.github.io/rocket-league-gym/).",
        "Complements the tutorial mirror in `docs/rlgym-site/`.",
        "Re-run `scripts/scrape_rlgym_api_docs.py` to refresh.",
        "",
        "## Pages",
        "",
    ]
    for title, rel in index_rows:
        lines.append(f"- [{title}]({rel})")
    lines.append("")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(index_rows)} pages to {OUT_DIR}")


if __name__ == "__main__":
    main()
