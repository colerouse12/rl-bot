# RocketSim

The project uses two separate simulator builds. The Python `rocketsim==2.2.1`
binding powers the RLGym reference checks. The C++ trainer uses RocketSim bundled
in the pinned GigaLearn source, identified by Git tree
`43a3765dc8a778f7f4b9168b7db5af94b12767e0` (native runtime reports 2.1.1).
Training is fixed to standard soccar geometry with one blue and one orange car
per arena. These versions are not interchangeable
policy contracts.

## Python reference

```powershell
.\.venv\Scripts\python.exe -m rl_bot.check_rocketsim
```

The reference check passes with Python 3.11.16, RLGym 2.0.1 and RocketSim 2.2.1.
It exercises 1v1 kickoff/reset, 92 finite float32 observations, 90 actions, car
movement, floor/wall collisions, goal detection and eight-tick action repetition.
The Python observation contract is not used by the C++ learner.

## Native training checks

After building, run:

```powershell
.\scripts\train.ps1 -Config configs/1v1-smoke.json -ContractCheck
.\scripts\train.ps1 -Config configs/1v1-smoke.json -CheckEnvironment
```

The mesh-free contract check exercises C++ AdvancedObs (109 values), action
indices/masks and team inversion. The native environment check exercises actual
soccar physics: one blue/one orange player, kickoff/reset, floor/wall collisions,
old controls for seven ticks followed by new controls for one tick, goal reward
signs, goal termination, no-touch truncation and the maximum episode timer.
The complete training acceptance check also performs optimizer updates and resume;
see [training](training-and-checkpoints.md).

Both native checks passed on 2026-09-16 with observation schema v2, including the
corrected boost-pad timer perspective and asymmetric cooldown checks for both teams.

## Local collision meshes

The launcher accepts `-MeshDirectory PATH`, where PATH contains `soccar/*.cmf`.
It first looks for ignored repository `collision_meshes/`. If absent, it uses the
existing mesh directory installed with the RLGym Python support package:

`.venv/Lib/site-packages/rlgym/rocket_league/sim/collision_meshes`

On this machine all 16 nonempty soccar meshes are present there. A static parse of
the mesh format and hashes exactly matched the pinned native RocketSim expected
soccar mesh set. This establishes asset compatibility; native physics checks are
the separate runtime evidence. No assets need to be copied into tracked files.
Collision meshes and proprietary game assets must remain outside version control.

## Runtime and deployment boundary

GigaLearn and RLGymCPP link statically into the project executable so the entry
point and learner share one RocketSim initialization state and thread pool.
The 109-value observation schema v2, 90-action ordering, inversion and tick/delay
contract are defined in `training/contract.h`. A future RLBot adapter must reuse
that implementation and validate live-game parity. The current Python checks do
not establish RLBot deployment readiness.
