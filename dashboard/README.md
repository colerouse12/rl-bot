# Local training dashboard

From the repository root, launch:

```powershell
.\.venv\Scripts\python.exe dashboard\server.py
```

Open <http://127.0.0.1:8765/>. The server is read-only and loopback-only. It
polls `/api/state` every three seconds, reads the active run pointer and status,
incrementally reads `METRICS.jsonl` (with lowercase compatibility), and reports numeric
checkpoint directories that contain `PROJECT_METADATA.json`. Metrics are
training diagnostics, not evaluated win rate or game strength. It accepts
partial JSONL writes and finite-value filtering, and never follows paths outside
the repository. The progress cards show cumulative transitions, target,
remaining work, and an ETA derived from fresh session telemetry; an ETA is
withheld until resumed runs produce matching session records. The scoring panel shows reconstructed combined goals and
timeout reset rates normalized to simulated five-minute arena time; these are
training episode rates, not completed-match averages or win rate. There are no
start, stop, delete, or external network requests.

The Python server caches its file offset, scoring totals, the last 100 scoring
records, and chart history (up to 2,400 records / 4 MiB of source JSONL). It reads
history once after startup and parses only newly appended complete records on
later refreshes. Totals do not shrink when old chart points fall out of the window.
Partial lines wait for the next refresh. File replacement, shrinking, or a new
metrics path resets the cache. Checkpoint/status metadata are still read on refresh.
The first record and last cached record are checked to detect in-place rewrites,
including a replacement of equal or greater size.

For the initial run, goals are reconstructed exactly from goal-event frequency
times simulated arena steps. Timeout rates are explicitly estimated using mean
episode length; those logs did not include direct timeout counts. New trainer
builds emit direct goal, timeout and simulated-time counters. A five-minute rate
uses `event_count * 300 / summed_simulated_arena_seconds`, across all arenas.
Recent rates use the last 100 updates, not the last five wall-clock minutes.

This reporting does not change rewards or decisions. Its CPU/disk/serialization
overhead is not zero and has not been benchmarked against trainer throughput.
