import json
from pathlib import Path
import tempfile
import unittest

from dashboard.scoring import ScoringHistory, scoring_record


def record(iteration=1, exact=False):
    # 900 player transitions = 450 arena steps = 30 simulated seconds.
    metrics = {"Collected Timesteps": 900, "Game/Goal": 2 / 450, "Episode Length": 90}
    if exact:
        metrics.update({"Game/Goals": 2, "Game/Timeouts": 3, "Game/Simulated Seconds": 30})
    return {"total_iterations": iteration, "metrics": metrics}


class ScoringTests(unittest.TestCase):
    def test_historical_counts_and_explicit_estimate(self):
        value = scoring_record(record())
        self.assertEqual(value["goals"], 2)
        self.assertEqual(value["seconds"], 30)
        self.assertEqual(value["timeouts"], 3)
        self.assertEqual(value["method"], "estimated")

    def test_exact_timeout_counter_takes_precedence(self):
        row = record(exact=True)
        row["metrics"]["Episode Length"] = 1
        self.assertEqual(scoring_record(row)["timeouts"], 3)
        self.assertEqual(scoring_record(row)["method"], "exact")

    def test_append_partial_line_no_double_count_and_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.jsonl"
            history = ScoringHistory()
            path.write_text(json.dumps(record()) + "\n", encoding="utf-8")
            first = history.read(path)
            self.assertEqual(first["total_goals"], 2)
            self.assertEqual(first["goals_per_5_minutes"], 20)
            self.assertEqual(first["timeouts_per_5_minutes"], 30)
            self.assertEqual(history.read(path), first)
            second = json.dumps(record(2, exact=True)).encode()
            with path.open("ab") as stream:
                stream.write(second[:20])
            self.assertEqual(history.read(path), first)
            with path.open("ab") as stream:
                stream.write(second[20:] + b"\n")
            result = history.read(path)
            self.assertEqual(result["total_goals"], 4)
            self.assertEqual(result["timeout_method"], "mixed")
            self.assertEqual(len(history.metrics()), 2)
            path.write_text(json.dumps(record(3)) + "\n", encoding="utf-8")
            self.assertEqual(history.read(path)["total_goals"], 2)

    def test_full_totals_survive_bounded_chart_and_recent_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.jsonl"
            path.write_text("".join(json.dumps(record(i, exact=True)) + "\n" for i in range(1, 2402)), encoding="utf-8")
            history = ScoringHistory()
            result = history.read(path)
            self.assertEqual(result["total_goals"], 4802)
            self.assertEqual(result["recent_goals"], 200)
            self.assertEqual(result["recent_iterations"], 100)
            self.assertEqual(len(history.metrics()), 2400)
            self.assertEqual(history.metrics()[0]["total_iterations"], 2)

    def test_same_size_and_larger_rewrite_resets_history(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.jsonl"
            history = ScoringHistory()
            line = lambda i: json.dumps(record(i, exact=True)) + "\n"
            path.write_text(line(1) + line(2), encoding="utf-8")
            history.read(path)
            # Same prefix and file identity, but a replaced cached boundary.
            path.write_text(line(1) + line(3), encoding="utf-8")
            self.assertEqual(history.read(path)["last_goal_iteration"], 3)
            self.assertEqual(len(history.metrics()), 2)
            path.write_text(line(4) + line(5) + line(6), encoding="utf-8")
            self.assertEqual(history.read(path)["total_goals"], 6)
            self.assertEqual(history.metrics()[0]["total_iterations"], 4)


if __name__ == "__main__":
    unittest.main()
