"""Supervise a timed or cumulative-transition standard-map 1v1 training session.

Run with the repository's Python 3.11 environment. The native trainer controls
its own stopping limit and saves after the final iteration; this process records its
status and captures the console log for the local dashboard and follow-up.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def idle_sleep_request(enabled: bool) -> bool:
    """Keep Windows awake only for this supervisor's lifetime; allow display sleep."""
    if os.name != "nt":
        return False
    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    set_state.argtypes = [ctypes.c_uint32]
    set_state.restype = ctypes.c_uint32
    if not set_state(0x80000000 | (0x00000001 if enabled else 0)):
        raise RuntimeError("Could not update the Windows idle-sleep request")
    return enabled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--seconds", type=int)
    limit.add_argument("--target-timesteps", type=int)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/1v1-cpu.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/1v1-cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-active-run", action="store_true", help="do not change the dashboard's active-run pointer (isolated checks)")
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        parser.error("run-id must contain only letters, numbers, underscores and hyphens")
    if args.seconds is None and args.target_timesteps is None:
        args.seconds = 1200
    if args.seconds is not None and args.seconds <= 0:
        parser.error("seconds must be positive")
    if args.target_timesteps is not None and not 0 < args.target_timesteps <= 9223372036854775807:
        parser.error("target-timesteps must be a positive signed 64-bit integer")
    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_file = run_dir / "status.json"
    if status_file.exists():
        parser.error("This run-id already has a status file; use a fresh run-id")
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"
    status = {
        "run_id": args.run_id, "status": "starting", "started_at": utc_now(),
        "finished_at": None, "pid": None, "supervisor_pid": os.getpid(),
        "checkpoint_dir": str(checkpoint_dir), "log_path": str(log_path),
        "config_path": str(args.config.resolve()), "duration_seconds": args.seconds,
        "target_timesteps": args.target_timesteps, "resumed_from_timesteps": 0,
        "resume": args.resume, "exit_code": None,
        "scope": "standard-map soccar 1v1; eight independent arenas in the CPU configuration",
    }
    write_json(status_file, status)
    lock_path = checkpoint_dir / ".training-session.lock"
    locked = False
    process = None
    sleep_requested = False
    try:
        # Refuse a second supervisor writing this run; stale locks are left for
        # explicit inspection instead of guessing whether another trainer exists.
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        locked = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock:
            json.dump({"run_id": args.run_id, "supervisor_pid": os.getpid()}, lock)
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        status["num_games"] = config["num_games"]
        status["run_config"] = config
        if args.resume:
            saved = sorted((p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.isdecimal()), key=lambda p: int(p.name))
            if not saved:
                raise RuntimeError("Resume requires an existing checkpoint")
            metadata = json.loads((saved[-1] / "PROJECT_METADATA.json").read_text(encoding="utf-8-sig"))
            status["resumed_from_timesteps"] = metadata["total_timesteps"]
            status["resumed_from_iterations"] = metadata["total_iterations"]
        if args.target_timesteps is not None and args.target_timesteps <= status["resumed_from_timesteps"]:
            raise RuntimeError("The cumulative timestep target has already been reached")
        if not args.no_active_run:
            write_json(ROOT / "runs/active-training.json", {"run_id": args.run_id, "status_file": str(status_file)})
        meshes = ROOT / "collision_meshes"
        if not (meshes / "soccar").is_dir():
            meshes = ROOT / ".venv/Lib/site-packages/rlgym/rocket_league/sim/collision_meshes"
        command = [str(ROOT / "build/bin/Release/rl-bot-train.exe"), "--config", str(args.config.resolve()),
                   "--checkpoint-dir", str(checkpoint_dir), "--mesh-dir", str(meshes),
                   "--max-iterations", "0"]
        if args.target_timesteps is not None:
            command.extend(["--target-timesteps", str(args.target_timesteps)])
        else:
            command.extend(["--max-seconds", str(args.seconds)])
        if args.resume:
            command.append("--resume")
        status["command"] = command
        environment = os.environ.copy()
        environment["PYTHONHOME"] = sys.base_prefix
        environment["PYTHONPATH"] = str(ROOT / ".venv/Lib/site-packages")
        environment["PATH"] = sys.base_prefix + os.pathsep + environment.get("PATH", "")
        sleep_requested = idle_sleep_request(True)
        status["prevents_idle_sleep"] = sleep_requested
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            status.update(status="running", pid=process.pid, started_at=utc_now())
            write_json(lock_path, {"run_id": args.run_id, "supervisor_pid": os.getpid(), "pid": process.pid})
            write_json(status_file, status)
            try:
                exit_code = process.wait(timeout=args.seconds + 300 if args.seconds is not None else None)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=30)
                raise RuntimeError("Trainer exceeded the requested duration plus the five-minute shutdown allowance")
        status["exit_code"] = exit_code
        if exit_code != 0:
            raise RuntimeError(f"Trainer exited with code {exit_code}; inspect training.log")
        saved = sorted((p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.isdecimal()), key=lambda p: int(p.name))
        if not saved:
            raise RuntimeError("Trainer exited without a checkpoint")
        latest = saved[-1]
        required_files = ("PROJECT_METADATA.json", "RUNNING_STATS.json", "POLICY.lt", "CRITIC.lt", "SHARED_HEAD.lt",
                          "POLICY_OPTIM.lt", "CRITIC_OPTIM.lt", "SHARED_HEAD_OPTIM.lt")
        if any(not (latest / name).is_file() or (latest / name).stat().st_size == 0 for name in required_files):
            raise RuntimeError("Final checkpoint is incomplete")
        metadata = json.loads((latest / "PROJECT_METADATA.json").read_text(encoding="utf-8-sig"))
        counters = json.loads((latest / "RUNNING_STATS.json").read_text(encoding="utf-8-sig"))
        if (metadata["total_timesteps"] != int(latest.name) or
                any(metadata[key] != counters[key] for key in ("total_timesteps", "total_iterations"))):
            raise RuntimeError("Final checkpoint counters disagree")
        if metadata["resumed_from_timesteps"] != status["resumed_from_timesteps"]:
            raise RuntimeError("Final checkpoint does not match the resume origin")
        if metadata.get("target_timesteps") != args.target_timesteps:
            raise RuntimeError("Final checkpoint does not match the requested target")
        expected_reason = "timestep-target" if args.target_timesteps is not None else "time-limit"
        if metadata["save_reason"] != expected_reason:
            raise RuntimeError(f"Unexpected final save reason: {metadata['save_reason']}")
        if args.seconds is not None and metadata["elapsed_training_seconds"] < args.seconds:
            raise RuntimeError("Training stopped before the requested duration")
        if args.target_timesteps is not None and metadata["total_timesteps"] < args.target_timesteps:
            raise RuntimeError("Training stopped before the cumulative transition target")
        status.update(status="completed", latest_checkpoint=str(latest),
                      total_timesteps=metadata["total_timesteps"], total_iterations=metadata["total_iterations"],
                      elapsed_training_seconds=metadata["elapsed_training_seconds"],
                      final_metrics=metadata["metrics"])
    except BaseException as exception:
        status.update(status="failed", error=str(exception) or type(exception).__name__)
        if process is not None:
            status["exit_code"] = process.poll()
    finally:
        status["finished_at"] = utc_now()
        child_alive = process is not None and process.poll() is None
        if child_alive:
            status["child_still_running"] = True
        try:
            write_json(status_file, status)
        finally:
            if locked and not child_alive:
                lock_path.unlink(missing_ok=True)
            if sleep_requested:
                idle_sleep_request(False)
    print(json.dumps(status, indent=2), flush=True)
    if status["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
