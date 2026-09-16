# Rocket League 1v1 bot

A CPU training setup for **standard-map soccar 1v1s** using GigaLearnCPP, RLGymCPP
and RocketSim, with a versioned
C++ observation/action contract and verified checkpoint loading. The Python
RLGym environment is a separate reference harness. RLBot v5 deployment and
fixed-opponent evaluation are later milestones.

Training readiness passed on 2026-09-16: native physics and contract checks,
real optimizer updates, checkpoint resume, retention and an eight-arena update.
Each parallel arena is a separate 1v1 with one blue and one orange car.

Use **Python 3.11 in `.venv`**. Setup and prerequisites are documented in
[environment setup](docs/project/setup.md).

```powershell
# Build and run bounded acceptance checks.
.\scripts\build-training.ps1
.\.venv\Scripts\python.exe scripts/verify_training.py

# Start eight independent standard-map 1v1 arenas on CPU.
.\scripts\train.ps1

# Continue its latest checkpoint.
.\scripts\train.ps1 -Resume
```

Ctrl+C requests a save and exit after the current PPO iteration. Use
`-MaxIterations 10 -CheckpointDirectory checkpoints/my-experiment` for a bounded
run. Existing checkpoints require `-Resume`; incompatible or incomplete saves
are rejected. The [training workflow](docs/project/training-and-checkpoints.md)
describes settings, contract, retention and validation results.

Native readiness must be established by `verify_training.py`, which performs
real optimizer updates and fresh-process resume. Its generated checkpoints and
logs are isolated under ignored `artifacts/training/`. The normal training run
uses ignored `checkpoints/1v1-cpu/`.

The [documentation index](docs/README.md) links setup, simulator details, source
references and the remaining development milestones. No third-party source,
collision assets, binaries or generated models are committed.
