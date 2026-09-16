"""Run native contract, optimizer, save/resume and retention acceptance checks.

Run with .venv/Scripts/python.exe. Every run creates a separate ignored output
directory and never deletes or changes an existing training run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoints(directory: Path) -> list[Path]:
    return sorted((p for p in directory.iterdir() if p.is_dir() and p.name.isdecimal()), key=lambda p: int(p.name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", type=Path, default=ROOT / "build/bin/Release/rl-bot-train.exe")
    parser.add_argument("--mesh-dir", type=Path, default=ROOT / ".venv/Lib/site-packages/rlgym/rocket_league/sim/collision_meshes")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    require(args.trainer.is_file(), "Build the trainer with scripts/build-training.ps1 first")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = ROOT / "artifacts/training" / run_id
    output.mkdir(parents=True)
    smoke_dir = output / "smoke-checkpoints"
    env = os.environ.copy()
    env["PYTHONHOME"] = sys.base_prefix
    env["PYTHONPATH"] = str(ROOT / ".venv/Lib/site-packages")
    env["PATH"] = sys.base_prefix + os.pathsep + env.get("PATH", "")
    results = []

    def run(name: str, *extra: str, success: bool = True, expected_error: str = "", config: str = "configs/1v1-smoke.json") -> str:
        command = [str(args.trainer.resolve()), "--config", config, "--mesh-dir", str(args.mesh_dir.resolve()), *map(str, extra)]
        result = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", timeout=args.timeout)
        (output / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        require((result.returncode == 0) == success, f"{name} returned {result.returncode}; see {output / (name + '.log')}")
        if expected_error:
            require(expected_error.lower() in result.stdout.lower(), f"{name} failed for the wrong reason; see log")
        results.append({"check": name, "exit_code": result.returncode, "log": f"{name}.log"})
        print(f"PASS {name}", flush=True)
        return result.stdout

    run("contract", "--contract-check")
    run("config", "--validate-config")
    run("environment", "--check-environment")
    run("first-update", "--checkpoint-dir", smoke_dir, "--max-iterations", "1")
    first = checkpoints(smoke_dir)[-1]
    first_stats = read_json(first / "RUNNING_STATS.json")
    require(first_stats["total_iterations"] == 1 and first_stats["total_timesteps"] > 0, "No completed first update")
    first_hash = digest(first / "POLICY.lt")
    required_files = ["POLICY.lt", "POLICY_OPTIM.lt", "CRITIC.lt", "CRITIC_OPTIM.lt", "SHARED_HEAD.lt", "SHARED_HEAD_OPTIM.lt"]
    for name in required_files:
        require((first / name).stat().st_size > 0, f"Missing initial {name}")
    first_metadata = read_json(first / "PROJECT_METADATA.json")
    require(first_metadata["metrics"]["Optimizer Steps"] > 0, "Metadata does not prove an optimizer step")
    run("fresh-run-protection", "--checkpoint-dir", smoke_dir, success=False, expected_error="Checkpoints already exist")
    # Alter only a copied metadata file, leaving the verified checkpoint intact.
    mismatch_dir = output / "mismatched-checkpoints"
    copied = mismatch_dir / first.name
    shutil.copytree(first, copied)
    mismatch = read_json(copied / "PROJECT_METADATA.json")
    mismatch["compatibility"]["contract"]["observation_schema"] = "incompatible-test-schema"
    (copied / "PROJECT_METADATA.json").write_text(json.dumps(mismatch), encoding="utf-8")
    run("contract-mismatch-rejected", "--checkpoint-dir", mismatch_dir, "--resume", success=False, expected_error="contract mismatch")
    (copied / "PROJECT_METADATA.json").write_text(json.dumps(first_metadata), encoding="utf-8")
    # Rename one optimizer in the isolated copy to exercise fail-closed resume.
    (copied / "POLICY_OPTIM.lt").rename(copied / "POLICY_OPTIM.lt.test-backup")
    run("partial-checkpoint-rejected", "--checkpoint-dir", mismatch_dir, "--resume", success=False, expected_error="Incomplete checkpoint")
    resume_output = run("resume-and-retention", "--checkpoint-dir", smoke_dir, "--resume", "--max-iterations", "3")
    require("RESUME_VERIFIED" in resume_output, "Saved model and optimizer state were not verified after load")
    saved = checkpoints(smoke_dir)
    latest = saved[-1]
    stats = read_json(latest / "RUNNING_STATS.json")
    require(stats["total_iterations"] == 4, "Resume reset or failed to continue the iteration counter")
    require(stats["total_timesteps"] > first_stats["total_timesteps"], "Resume did not increase timesteps")
    require(digest(latest / "POLICY.lt") != first_hash, "Policy did not change across resumed updates")
    require(len(saved) == 3 and first not in saved, "Checkpoint retention did not keep the newest three saves")
    save_reasons = set()
    for checkpoint in saved:
        for name in required_files:
            require((checkpoint / name).stat().st_size > 0, f"Incomplete checkpoint: {checkpoint / name}")
        metadata = read_json(checkpoint / "PROJECT_METADATA.json")
        save_reasons.add(metadata["save_reason"])
        require(metadata["metrics"]["Optimizer Steps"] > 0, "Checkpoint has no optimizer-step evidence")
        require(metadata["resumed_from_timesteps"] == first_stats["total_timesteps"], "Resume provenance does not match")
    require(save_reasons == {"periodic-save", "bounded-run-limit"}, "Periodic and explicit final saves were not both verified")
    # Validate the actual multi-game launch configuration through one bounded update.
    run("multi-game-update", "--checkpoint-dir", output / "cpu-checkpoints", "--max-iterations", "1", config="configs/1v1-cpu.json")
    cpu_stats = read_json(checkpoints(output / "cpu-checkpoints")[-1] / "RUNNING_STATS.json")
    require(cpu_stats["total_iterations"] == 1 and cpu_stats["total_timesteps"] >= 8192, "Multi-game configuration did not complete its rollout")
    report = {"status": "PASS", "utc": datetime.now(timezone.utc).isoformat(),
              "trainer": str(args.trainer.resolve()), "trainer_sha256": digest(args.trainer),
              "checks": results, "first_timesteps": first_stats["total_timesteps"],
              "resumed_timesteps": stats["total_timesteps"], "resumed_iterations": stats["total_iterations"],
              "cpu_timesteps": cpu_stats["total_timesteps"],
              "retained_checkpoints": [p.name for p in saved]}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Training acceptance checks passed: {output / 'verification.json'}")


if __name__ == "__main__":
    main()
