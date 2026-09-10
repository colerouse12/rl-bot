# RocketSim setup and verification

This note owns native simulation prerequisites, timing and runtime checks. Package
links live in [official references](../REFERENCES.md); general environment creation
lives in [setup](setup.md).

RocketSim 2.2.1 is installed in the Python 3.11 `.venv` and imported as `RocketSim`.
RLGym supplies package-relative Soccar collision meshes, and `RocketSimEngine`
initializes the binding with that directory. No extra arena download is required on
this machine. Do not download or commit proprietary game assets.

Run:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m rl_bot.check_rocketsim
.\.venv\Scripts\python.exe -m rl_bot.smoke
```

Verified 2026-09-10 on Windows x64 / Python 3.11.16 / NumPy 1.26.4:

- dependency metadata is compatible;
- both cars move under controls and observations remain finite;
- the ball collides with the arena floor and side wall;
- goals terminate both agents and reward team perspectives correctly;
- repeated kickoff resets restore a valid 1v1 state; and
- the smoke test advances 128 physics ticks with two `[92]` float32 observations.

Training uses `RocketSimEngine(rlbot_delay=True)` and 90 discrete lookup-table
actions repeated for eight physics ticks. Deployment must preserve that timing.

If an import fails, confirm `.venv\Scripts\python.exe` is Python 3.11 and reinstall
from `requirements-lock.txt`. If arena creation or collision checks fail, verify
`.venv/Lib/site-packages/rlgym/rocket_league/sim/collision_meshes/soccar` exists.
A successful import alone does not prove that the arena or collision data works.

Python 3.11 matches RocketSim 2.2.1’s internal Windows wheel tag, so dependency
validation passes without editing package metadata or bypassing resolution.

Implementation: `rl_bot/check_rocketsim.py`, `rl_bot/smoke.py`, and
`rl_bot/environment.py`.
