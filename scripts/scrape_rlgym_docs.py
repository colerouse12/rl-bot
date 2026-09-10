"""Scrape instructional pages from https://rlgym.org into local markdown.

Source: public RLGym documentation site. Content belongs to RLGym authors;
kept locally for offline/project reference. Re-run to refresh.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import html2text

SITE = "https://rlgym.org"
SITEMAP = f"{SITE}/sitemap.xml"
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "rlgym-site"
UA = "rl-bot-local-docs/0.1 (offline instructional mirror; contact via project repo)"

SKIP_PREFIXES = (
    "/blog",
    "/markdown-page",
)

h2t = html2text.HTML2Text()
h2t.body_width = 0
h2t.ignore_images = False
h2t.ignore_links = False
h2t.protect_links = True
h2t.mark_code = False


def fetch(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def sitemap_urls() -> list[str]:
    root = ET.fromstring(fetch(SITEMAP))
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [el.text.strip() for el in root.findall("sm:url/sm:loc", ns) if el.text]
    kept = []
    for url in urls:
        path = urlparse(url).path.rstrip("/") or "/"
        if path == "/":
            continue
        if any(path.startswith(p) for p in SKIP_PREFIXES):
            continue
        kept.append(url if url.endswith("/") else url + "/")
    return sorted(set(kept))


def url_to_relpath(url: str) -> Path:
    path = unquote(urlparse(url).path).strip("/")
    if not path:
        return Path("index.md")
    return Path(path) / "index.md" if not path.endswith(".md") else Path(path)


def extract_article(html: bytes, url: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.select_one("h1") or soup.select_one("title")
    title = title_el.get_text(" ", strip=True) if title_el else url
    title = re.sub(r"\s*\|\s*RLGym\s*$", "", title).strip()

    article = soup.select_one("article .theme-doc-markdown") or soup.select_one(
        "article"
    )
    if article is None:
        raise RuntimeError(f"No article content found for {url}")

    # Drop nav/pagination chrome inside article if present
    for sel in [".pagination-nav", "nav", ".theme-doc-toc-mobile"]:
        for el in article.select(sel):
            el.decompose()

    md = h2t.handle(str(article)).strip()
    md = re.sub(r"\n{3,}", "\n\n", md)
    header = (
        f"# {title}\n\n"
        f"> Source: [{url}]({url})  \n"
        f"> Scraped for local reference from the public RLGym docs.\n\n"
    )
    return title, header + md + "\n"


def main() -> None:
    urls = sitemap_urls()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows: list[tuple[str, str]] = []

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        html = fetch(url)
        title, markdown = extract_article(html, url)
        rel = url_to_relpath(url)
        dest = OUT_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(markdown, encoding="utf-8")
        index_rows.append((title, rel.as_posix()))
        time.sleep(0.35)

    index_lines = [
        "# RLGym site docs (local mirror)",
        "",
        "Instructional pages scraped from [rlgym.org](https://rlgym.org/) for offline use.",
        "Re-run `scripts/scrape_rlgym_docs.py` to refresh.",
        "",
        "## Pages",
        "",
    ]
    for title, rel in index_rows:
        index_lines.append(f"- [{title}]({rel})")
    index_lines.append("")
    (OUT_DIR / "README.md").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"Wrote {len(index_rows)} pages to {OUT_DIR}")


if __name__ == "__main__":
    main()
