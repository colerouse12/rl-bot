# GigaLearn training and checkpoints

Production training uses the pinned GigaLearn C++ learner. The project entry point
is `training/main.cpp`; `training/contract.h` is the production observation/action
boundary. The Python `BotObs` remains a separate 92-value reference environment.

Scope is **standard-map soccar 1v1 only**: one blue and one orange car in each
arena. `num_games=8` runs eight independent 1v1 arenas in parallel. The environment
factory always creates `GameMode::SOCCAR`; team size is fixed in code and guarded
by the observation contract.

## Verified readiness: 2026-09-16

The Release CPU executable built successfully and both CTest checks passed
(startup and worker exception propagation/recovery). All nine native acceptance
checks passed with observation schema v2 and runtime patch v2:

- Contract, both-team boost timer/inversion, native floor/wall physics, controls,
  kickoff/reset, goal rewards and timeout checks.
- First optimizer update and complete checkpoint at 152 transitions.
- Fresh-process model/Adam/counter verification, then three more updates to
  608 transitions and four total iterations.
- Retention kept only checkpoints 304, 456 and 608; periodic and final saves
  carried metadata. Fresh-run overwrite, incompatible schema and missing
  optimizer checks all failed safely as expected.
- The actual CPU configuration completed a bounded update across eight 1v1
  arenas, collecting 9,664 transitions with nonzero policy and critic updates.

The machine-local report is
`artifacts/training/20260916T053025Z-1128ccf9/verification.json`; adjacent logs and
complete test checkpoints are ignored build artifacts. The normal training
directory is separate. Earlier v1 bring-up checkpoints were explicitly rejected
by the corrected executable. This proves training infrastructure readiness, not
policy playing strength or RLBot deployment readiness.

## Build and launch

Follow [setup](setup.md), then run from the repository root:

```powershell
.\scripts\build-training.ps1
.\.venv\Scripts\python.exe scripts/verify_training.py

# Start a new training run (defaults to configs/1v1-cpu.json).
.\scripts\train.ps1

# Continue the latest checkpoint of that run.
.\scripts\train.ps1 -Resume

# Or run a bounded experiment in its own checkpoint directory.
.\scripts\train.ps1 -MaxIterations 10 -CheckpointDirectory checkpoints/experiment-1

# Train for twenty minutes, finishing the current update before saving.
.\scripts\train.ps1 -MaxSeconds 1200
```

`train.ps1` supplies the embedded Python runtime and an explicit mesh path. It
prefers local `collision_meshes/`, then the meshes already installed with RLGym.
`-Config`, `-CheckpointDirectory`, `-MeshDirectory`, and `-MaxIterations` override
those settings. Relative paths are rooted at the repository by this wrapper.
Direct executable paths are relative to its current working directory.

The executable supports `--help`, `--validate-config`, `--contract-check`, and
`--check-environment`. The last check exercises native physics without training.
The other two checks can run without collision assets. Unknown configuration
fields and invalid dimensions, timing or PPO batch settings fail before learning.

`max_iterations` counts **additional completed PPO iterations in this process**;
zero runs until interrupted. Ctrl+C requests a save and exit after the current
iteration. Rollouts finish episodes and may overshoot `timesteps_per_iteration`.
Force-killing a process can lose the current iteration or interrupt a save.

`-MaxSeconds` / `--max-seconds` sets a positive wall-clock training duration.
Timing starts immediately before the learner loop, after initialization and model
loading. The current PPO update finishes before the trainer saves with
`save_reason=time-limit` and exits. Final metadata records
`duration_limit_seconds` and `elapsed_training_seconds`.

The checkpoint root also contains `METRICS.jsonl`, flushed after each completed
update. It records all finite report metrics, cumulative counters, per-process
elapsed training time and `session_*` counters. Resume appends to this file;
elapsed time resets for each process, while cumulative counters continue.

`scripts/run_training.py` supervises a timed run, captures its console log,
records atomic `runs/<run-id>/status.json` updates, and points
`runs/active-training.json` at the active run for local monitoring. It requires
the final time-limit checkpoint before reporting success and refuses a second
supervised writer in the same checkpoint directory.

For a monitored twenty-minute experiment, use a unique run ID and a new
checkpoint directory (or explicitly pass `--resume` for an existing run):

```powershell
.\.venv\Scripts\python.exe scripts/run_training.py --run-id training-example --seconds 1200 --checkpoint-dir checkpoints/training-example
```

The timed executable passed both CTest checks after rebuilding. A separate
three-second supervisor check completed on 2026-09-16 with exit code 0, 57
iterations, 8,784 transitions and a `time-limit` checkpoint after 3.092 seconds.
Evidence is in `runs/supervisor-check-20260916T0648/status.json`; the earlier
nine-check acceptance report predates the duration/telemetry addition.

The requested twenty-minute run, `training-20m-20260916T065026Z`, started at
2026-09-16 06:50:27 UTC using `configs/1v1-cpu.json` and
`checkpoints/1v1-cpu/`. Completion and policy-quality results are not yet claimed.

## Local training dashboard

Launch `.\.venv\Scripts\python.exe dashboard/server.py` and open
`http://127.0.0.1:8765/`. This Python-standard-library server and HTML/CSS/JavaScript
page read the supervised run's status, live metrics and checkpoint metadata.
The page refreshes every three seconds and displays counters, speed, reward,
critic loss and policy entropy. It binds to loopback and provides read-only
monitoring. The [dashboard guide](../../dashboard/README.md) owns launch options.

The dashboard was verified against the active eight-arena run on 2026-09-16:
its browser counters updated and the API returned the live metrics and latest
checkpoint. Metric history is bounded to the last 4 MiB / 2,400 records, so a
long run may show only its recent history. Incomplete JSONL records are skipped.
Reward/loss/entropy describe training behavior; evaluated win rates and playing
strength are not yet available.

## Versioned 1v1 contract

- One blue and one orange car, with seeded random kickoff resets.
- 120 Hz physics, `tick_skip=8`, `action_delay=7`.
- `rlbot.advanced-1v1.v2`: exactly 109 finite float observations from the pinned
  `RLGC::AdvancedObs`, with the corrected boost-pad timer perspective, guarded by
  the project `Observation` class.
- `rlbot.default-action-90.v1`: the pinned `DefaultAction` table and masks. The
  full ordered table and mask vectors are included in checkpoint metadata.
- Goal termination; configurable no-touch and maximum episode time truncation.
- CPU inference/training, FP32, Adam, ReLU, optional layer normalization, return
  standardization enabled, observation standardization disabled.

Feature offsets are: ball position/velocity/angular velocity (0-8), previous
controls in throttle/steer/pitch/yaw/roll/jump/boost/handbrake order (9-16), 34 boost
pads (17-50), self (51-79), opponent (80-108). Each player contributes position,
forward, up, world velocity, world angular velocity, local angular velocity,
local relative ball position, local relative ball velocity, boost, on-ground,
has-flip-or-jump, demoed and has-jumped. Position divides by 5000, velocity by
2300, angular velocity by 3, boost by 100. Unavailable boost pads use
`1/(1+seconds_remaining)`; available pads use 1.

Orange observations negate world x/y and use inverted boost-pad order. Local
features use the inverted car orientation; previous controls retain their order
and sign. `--contract-check` tests symmetric team inversion, feature offsets,
control ranges, action masks, invalid indices and nonfinite observations. The
archived `GetBoostPadTimers` getter selected the opposite perspective from pad
availability. The v2 contract corrects that getter and tests asymmetric cooldowns
for both teams; earlier bring-up checkpoints using v1 are incompatible.

Rewards follow the pinned example: air 0.25, facing ball 0.25, velocity to ball 4,
strong touch `(20,100)` 60, zero-sum ball velocity to goal 2, boost pickup 10,
boost conservation 0.2, zero-sum bump 20, zero-sum demo 80, goal 150. These are
initial shaping choices, not evidence of policy quality. Every reward term,
observation, model output, loss, gradient and updated parameter is checked for
finite values. Iteration reports must show optimizer steps and parameter changes.

## Configurations

`configs/1v1-smoke.json` uses one game, a 128-transition target and one optimizer
epoch, with a single additional iteration by default. `configs/1v1-cpu.json` uses
eight games, an 8192-transition target, 1024-sized minibatches and two epochs.
Both use 128x128 shared, policy and critic layers so the contract can be tested
without excessive memory. Remote metrics and rendering are disabled. Local
metrics and build details are saved with checkpoints.

## Checkpoints and resume

The learner creates numeric directories under its configured checkpoint folder.
Each complete save has `RUNNING_STATS.json`, `POLICY.lt`, `CRITIC.lt`,
`SHARED_HEAD.lt`, the three corresponding `*_OPTIM.lt` files, and
`PROJECT_METADATA.json`. Metadata includes the effective configuration, rewards,
seed scope, contract, dependency identities, compiler/build/LibTorch versions,
iteration metrics, timestep and iteration counts, and model/optimizer summaries.

New runs reject a directory containing numeric checkpoints unless `--resume` is
provided. Resume selects the highest canonical numeric directory, requires every
file to be present and nonempty, and validates schema, sources, observation/action
contract, network and LibTorch compatibility before constructing the learner.
After load, model dimensions and parameter summaries, Adam state shapes and steps,
and counters are verified before collecting more experience.

Resume restores model, Adam and running statistics. It does **not** restore exact
simulation states or the random-number trajectory, so it is not a bit-for-bit
continuation. Missing/corrupt latest checkpoints fail visibly instead of silently
resetting a model or optimizer. Preserve a damaged directory for diagnosis and
restore a known-good complete checkpoint into a separate run directory if needed.

Periodic saves and bounded/interrupt exits attach metadata before retention runs.
The smoke configuration retains three checkpoints; the CPU configuration retains
eight. Copy promoted models and all metadata to an ignored named model directory
before retention can remove them. Do not run two writers against one checkpoint
directory.

## Pinned-source adaptations

`dependencies/gigalearn.json` pins the author-published source. Its bundled
components are ordinary Git trees, not actual submodule gitlinks; their exact tree
hashes are recorded without inventing separate commit IDs. The ignored checkout
is fetched for local use, not vendored or redistributed.

`scripts/patch_gigalearn.py` applies narrow adaptations for bounded runs, metadata
and iteration hooks, finite checks, worker exception propagation, correct
multi-game event/reward indexing, boost-pad timer perspective, and cleanup.
Static GigaLearn linkage ensures
the entry point and learner share one RocketSim runtime and thread pool. The
patcher rejects unexpected source edits and untracked files; configuration checks
its result again. This project does not implement a separate PPO learner.

## Evaluation and deployment boundary

Training readiness is established by native updates and verified save/resume.
There is not yet an RLBot deployment adapter or fixed-opponent evaluation suite.
Neither a successful update nor a checkpoint demonstrates a competitive bot.
The future RLBot v5 adapter must reuse `training/contract.h`, the pinned action and
observation implementation, and matching network structure. Live-game parity,
evaluation, GPU support and reward tuning remain later milestones.
