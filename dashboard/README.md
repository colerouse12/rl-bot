# Local training dashboard

From the repository root, launch:

```powershell
.\.venv\Scripts\python.exe dashboard\server.py
```

Open <http://127.0.0.1:8765/>. The server is read-only and loopback-only. It
polls `/api/state` every three seconds, reads the active run pointer and status,
tails `METRICS.jsonl` (with lowercase compatibility), and reports numeric
checkpoint directories that contain `PROJECT_METADATA.json`. Metrics are
training diagnostics, not evaluated win rate or game strength. It accepts
partial JSONL writes and finite-value filtering, and never follows paths outside
the repository. There are no start, stop, delete, or external network requests.
