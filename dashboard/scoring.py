"""Aggregate scoring rates without truncating totals to the chart history window."""
from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import threading


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def scoring_record(row: dict, tick_skip: int = 8) -> dict | None:
    metrics = row.get("metrics", {})
    collected = metrics.get("Collected Timesteps")
    if not _number(collected) or collected <= 0:
        return None
    # The fixed contract has two learning players per arena. Collected Timesteps
    # counts newly simulated player steps, unlike the completed trajectory size.
    arena_steps = collected / 2
    seconds = metrics.get("Game/Simulated Seconds", arena_steps * tick_skip / 120)
    if not _number(seconds) or seconds <= 0:
        return None
    goals = metrics.get("Game/Goals")
    if goals is None:
        goal_rate = metrics.get("Game/Goal")
        if not _number(goal_rate) or not 0 <= goal_rate <= 1:
            return None
        goals = goal_rate * arena_steps
    if not _number(goals) or goals < 0 or abs(goals - round(goals)) > 1e-6 * max(1, goals):
        return None
    exact_timeouts = metrics.get("Game/Timeouts")
    if _number(exact_timeouts) and exact_timeouts >= 0:
        timeouts = exact_timeouts
        method = "exact"
    else:
        episode_length = metrics.get("Episode Length")
        # Historical logs omit completed-episode counts and completed-trajectory
        # size. Exposure / mean episode length is only an estimate: episodes can
        # cross update boundaries, and PPO truncations can differ from env resets.
        timeouts = max(0, arena_steps / episode_length - round(goals)) if _number(episode_length) and episode_length > 0 else None
        method = "estimated" if timeouts is not None else "unavailable"
    return {"goals": round(goals), "seconds": seconds, "timeouts": timeouts,
            "method": method, "iteration": row.get("total_iterations")}


class ScoringHistory:
    def __init__(self):
        self.lock = threading.Lock()
        self.path = None
        self.identity = None
        self.offset = 0
        self.reset()

    def reset(self):
        self.offset = 0
        self.prefix = None
        self.boundary = b""
        self.goals = 0
        self.seconds = 0.0
        self.timeouts = 0.0
        self.iterations = 0
        self.last_goal = None
        self.methods = set()
        self.recent = deque(maxlen=100)
        self.chart_rows = deque()
        self.chart_bytes = 0
        self.complete = True

    def read(self, path: Path, tick_skip: int = 8) -> dict | None:
        with self.lock:
            try:
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if self.path != path or self.identity != identity or stat.st_size < self.offset:
                    self.reset()
                    self.path, self.identity = path, identity
                with path.open("rb") as stream:
                    prefix = stream.readline()
                    stream.seek(max(0, self.offset - len(self.boundary)))
                    boundary = stream.read(len(self.boundary))
                    if ((self.prefix is not None and prefix != self.prefix) or
                            (self.boundary and boundary != self.boundary)):
                        self.reset()
                    if prefix.endswith(b"\n"):
                        self.prefix = prefix
                    stream.seek(self.offset)
                    while line := stream.readline():
                        if not line.endswith(b"\n"):
                            break
                        self.offset = stream.tell()
                        self.boundary = line
                        try:
                            row = json.loads(line)
                            if isinstance(row, dict):
                                self.chart_rows.append((row, len(line)))
                                self.chart_bytes += len(line)
                                while len(self.chart_rows) > 2400 or self.chart_bytes > 4 * 1024 * 1024:
                                    self.chart_bytes -= self.chart_rows.popleft()[1]
                            item = scoring_record(row, tick_skip) if isinstance(row, dict) else None
                        except (ValueError, TypeError, AttributeError):
                            item = None
                        if item is None:
                            self.complete = False
                            continue
                        self.goals += item["goals"]
                        self.seconds += item["seconds"]
                        self.timeouts += item["timeouts"] or 0
                        self.methods.add(item["method"])
                        self.iterations += 1
                        if item["goals"]:
                            self.last_goal = item["iteration"]
                        self.recent.append(item)
            except OSError:
                return None
            if not self.iterations:
                return None
            recent_seconds = sum(r["seconds"] for r in self.recent)
            recent_goals = sum(r["goals"] for r in self.recent)
            recent_timeouts = sum(r["timeouts"] or 0 for r in self.recent)
            timeout_method = next(iter(self.methods)) if len(self.methods) == 1 else "mixed"
            recent_methods = {r["method"] for r in self.recent}
            recent_timeout_method = next(iter(recent_methods)) if len(recent_methods) == 1 else "mixed"
            note = ("Directly counted episode resets without a goal." if timeout_method == "exact" else
                    "Estimated from mean episode length and simulated time, minus goals. Historical logs did not count timeout resets; update boundaries and PPO truncations can affect this estimate.")
            return {
                "total_goals": self.goals, "goals_per_5_minutes": self.goals * 300 / self.seconds,
                "recent_goals": recent_goals, "recent_goals_per_5_minutes": recent_goals * 300 / recent_seconds,
                "recent_iterations": len(self.recent), "simulated_arena_seconds": self.seconds,
                "iterations": self.iterations, "last_goal_iteration": self.last_goal,
                "source": "counted-or-reconstructed-step-events", "complete_history": self.complete,
                "timeout_count": self.timeouts,
                "timeouts_per_5_minutes": None if "unavailable" in self.methods else self.timeouts * 300 / self.seconds,
                "recent_timeouts_per_5_minutes": None if any(r["timeouts"] is None for r in self.recent) else recent_timeouts * 300 / recent_seconds,
                "timeout_method": timeout_method, "recent_timeout_method": recent_timeout_method,
                "timeout_note": note,
            }

    def metrics(self) -> list[dict]:
        with self.lock:
            return [row for row, _ in self.chart_rows]
