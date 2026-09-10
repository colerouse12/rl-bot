# Environment setup

This note owns interpreter selection, dependency installation and compatibility
checks. Simulator details belong in [RocketSim](rocketsim.md); external links belong
in [official references](../REFERENCES.md).

## Supported local environment

Use 64-bit Python **3.11** in `.venv`. This is the common supported interpreter for
RLBot 2.0.0b52, rlgym-rlbot 0.3.0, RLGym Learn 2.0.0, PyTorch 2.14.0 and RocketSim’s
Windows wheel. Python 3.11.16 is installed here.

To recreate the environment after installing Python 3.11 from python.org and
confirming `py -3.11 --version` works:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Alternatively, use uv when Python 3.11 is not registered with the Windows launcher:

```powershell
uv python install 3.11
uv venv --seed --python 3.11 .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The current environment was created with uv because this machine's `py` launcher
does not list Python 3.11. Do not copy the uv-managed interpreter's absolute path
into project configuration; `.venv\Scripts\python.exe` is the stable project path.

`requirements.txt` lists direct project dependencies. `requirements-lock.txt`
captures the complete set verified together. Update both deliberately when changing
the stack; never use `--no-deps` to conceal incompatibilities.

Activate with `.\.venv\Scripts\Activate.ps1`, or select
`.venv\Scripts\python.exe` directly in the editor.

## Verification

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m rl_bot.smoke
.\.venv\Scripts\python.exe -m rl_bot.check_rocketsim
```

Verified 2026-09-10 on Windows x64 / Python 3.11.16:

- `pip check`: no broken requirements.
- Learner, algorithms, PyTorch and RLGym-RLBot bridge imports: pass.
- 1v1 simulator smoke and native RocketSim physics checks: pass.

This confirms the architecture dependencies and simulator boundary. Training and
deployment are later implementation milestones, so their absence is not a setup
failure.
