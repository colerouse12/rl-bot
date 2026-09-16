"""Local, read-only dashboard for a running 1v1 CPU training session.

Run from the repository root with ``python dashboard/server.py``.  The server
deliberately uses only the standard library and binds to loopback.
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from scoring import ScoringHistory

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = Path(__file__).resolve().parent
MAX_METRIC_BYTES = 4 * 1024 * 1024
MAX_METRIC_ROWS = 2400
SCORING_HISTORY = ScoringHistory()


def _finite(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_finite(item) for item in value[:128]]
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    return value if value is None or isinstance(value, str) else str(value)


def _json(path: Path):
    if not _under_root(path):
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            return _finite(json.load(stream))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _under_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def _bounded_text(path: Path, limit=MAX_METRIC_BYTES) -> str:
    try:
        with path.open("rb") as stream:
            data = stream.read(limit)
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _tail_text(path: Path, limit=MAX_METRIC_BYTES) -> str:
    """Read at most limit bytes ending at EOF, dropping its partial first line."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit), os.SEEK_SET)
            data = stream.read(limit)
        text = data.decode("utf-8", errors="replace")
        if size > limit:
            text = text.split("\n", 1)[-1]
        return text
    except OSError:
        return ""


def _metric_path(checkpoint_dir: Path) -> Path | None:
    for name in ("METRICS.jsonl", "metrics.jsonl"):
        candidate = checkpoint_dir / name
        if candidate.is_file() and _under_root(candidate):
            return candidate
    return None


def read_metrics(checkpoint_dir: Path) -> list[dict]:
    path = _metric_path(checkpoint_dir)
    if path is None:
        return []
    # Reading from the end keeps a long 20-minute run responsive.  A partial
    # final line is naturally ignored by the JSON decoder.
    text = _tail_text(path)
    rows = []
    for line in text.splitlines()[-MAX_METRIC_ROWS:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(_finite(item))
    return rows


def _status_candidates() -> list[Path]:
    runs = ROOT / "runs"
    if not runs.is_dir():
        return []
    return sorted(runs.glob("training-*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_status() -> dict | None:
    pointer = _json(ROOT / "runs" / "active-training.json")
    candidates = []
    if isinstance(pointer, dict):
        status_file = pointer.get("status_file")
        if isinstance(status_file, str):
            candidate = Path(status_file)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            if _under_root(candidate) and candidate.name == "status.json":
                candidates.append(candidate)
    candidates.extend(_status_candidates())
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        data = _json(candidate)
        if isinstance(data, dict):
            data["status_file"] = str(candidate.relative_to(ROOT))
            return data
    return None


def checkpoint_info(checkpoint_dir: Path) -> dict:
    result = {"directory": str(checkpoint_dir.relative_to(ROOT)), "latest": None, "checkpoints": []}
    if not checkpoint_dir.is_dir() or not _under_root(checkpoint_dir):
        return result
    for child in checkpoint_dir.iterdir():
        try:
            child = child.resolve()
        except OSError:
            continue
        if not _under_root(child) or not child.is_dir() or not child.name.isdigit():
            continue
        metadata_path = child / "PROJECT_METADATA.json"
        if not metadata_path.is_file() or not _under_root(metadata_path):
            continue
        metadata = _json(metadata_path)
        if metadata is None:
            continue
        result["checkpoints"].append({
            "id": int(child.name),
            "directory": str(child.relative_to(ROOT)),
            "metadata": metadata,
        })
    result["checkpoints"].sort(key=lambda item: item["id"])
    if result["checkpoints"]:
        result["latest"] = result["checkpoints"][-1]
    return result


def snapshot() -> dict:
    status = load_status()
    if not status:
        return {"run": None, "checkpoint": None, "metrics": [], "scoring": None, "server": {"root": str(ROOT)}}
    raw_dir = status.get("checkpoint_dir")
    directory = Path(raw_dir) if isinstance(raw_dir, str) else Path("checkpoints/1v1-cpu")
    if not directory.is_absolute():
        directory = ROOT / directory
    if not _under_root(directory):
        directory = ROOT / "checkpoints/1v1-cpu"
    info = checkpoint_info(directory.resolve())
    metric_file = _metric_path(directory.resolve())
    scoring = SCORING_HISTORY.read(metric_file, status.get("run_config", {}).get("tick_skip", 8)) if metric_file else None
    return {
        "run": status,
        "checkpoint": info,
        "metrics": [_finite(row) for row in SCORING_HISTORY.metrics()] if metric_file else [],
        "scoring": scoring,
        "server": {"root": str(ROOT), "metric_file": str(metric_file.relative_to(ROOT)) if metric_file else None},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "rl-bot-dashboard/1.0"

    def _send(self, status, content_type, body):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send(200, "application/json; charset=utf-8", json.dumps(snapshot(), separators=(",", ":")))
            return
        if path == "/api/health":
            self._send(200, "application/json; charset=utf-8", '{"ok":true}')
            return
        if path == "/" or path == "/index.html":
            self._static("index.html")
            return
        if path.startswith("/static/"):
            self._static(path.removeprefix("/static/"))
            return
        self._send(404, "text/plain; charset=utf-8", "Not found\n")

    def _static(self, name):
        candidate = (DASHBOARD / name).resolve()
        if not _under_dashboard(candidate) or not candidate.is_file():
            self._send(404, "text/plain; charset=utf-8", "Not found\n")
            return
        self._send(200, mimetypes.guess_type(candidate.name)[0] or "application/octet-stream", _bounded_text(candidate, 1024 * 1024))

    def log_message(self, fmt, *args):
        return


def _under_dashboard(path: Path) -> bool:
    try:
        path.relative_to(DASHBOARD.resolve())
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Serve the local Rocket League training dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="loopback host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("dashboard is localhost-only; use 127.0.0.1")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"rl-bot dashboard: http://{args.host}:{args.port}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
