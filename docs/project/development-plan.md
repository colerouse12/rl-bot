# Development plan

The target is a GigaLearnCPP-trained bot for **standard-map soccar 1v1s** that can
later run through RLBot v5. Every arena has one blue and one orange player;
parallel game count means independent 1v1 arenas.
GigaLearn supplies the C++ PPO learner; the Python RLGym 2.0.1 environment remains
a separate physics/behavior reference.

## Current status

Milestones 1-3 passed on 2026-09-16. The Release CPU trainer builds and passes
native contract/physics checks, optimizer updates, fresh-process resume,
checkpoint retention, rejection checks and an update across eight parallel 1v1
arenas. The corrected observation schema is `rlbot.advanced-1v1.v2`.
A twenty-minute CPU training run completed on 2026-09-16 using eight arenas:
12,464,976 transitions, 1,475 iterations, a verified time-limit checkpoint and
1,366 combined goals. The dashboard shows goals and estimated historical timeouts
per five minutes of simulated arena time. This establishes scoring activity,
not competitive playing strength. The normal training command is
`.\scripts\train.ps1`; see the training note for evidence and launch details.

## Milestones 1-3: training readiness

- Pin the author-published source at
  `86e11c3881a21225a43c36b14af40f0b3c87e7dd` and bundled dependency tree hashes.
  Fetch source into ignored storage for local use instead of vendoring it.
- Build a Release CPU executable with MSVC, CMake, Python 3.11 and CPU LibTorch.
- Define and test the shared 109-value observation / 90-action C++ contract,
  orange inversion, eight-tick steps and seven-tick action delay.
- Exercise native 1v1 kickoff/reset, floor/wall collisions, goals and timeouts.
- Complete real finite PPO updates with visible parameter changes.
- Save complete models, Adam optimizer states, counters, configuration and contract
  metadata; resume in a fresh process; verify retention and reject bad resumes.
- Validate one bounded update using the actual multi-game CPU launch configuration.

The acceptance command is `.\.venv\Scripts\python.exe scripts/verify_training.py`.
It isolates all validation checkpoints under ignored `artifacts/training/`.
Only a successful native report establishes these runtime milestones.

## Milestone 4: evaluation and model selection

Add deterministic and sampled evaluation against fixed opponents. Use fixed seeds
and opponent versions to compare checkpoints on goals, wins, touches, episode
length and inference latency independently of training reward. Preserve selected
checkpoints outside automatic retention. No competitive-performance claim follows
from infrastructure acceptance alone.

## Milestone 5: RLBot v5 deployment

Implement the C++ adapter using the same `training/contract.h`, pinned RLGymCPP
observation/action implementation and network layout. Verify blue/orange inversion,
kickoffs, eight-tick controls and missed-tick behavior in local Rocket League.
Load only complete models whose metadata matches the adapter's contract.

## Milestone 6: scaling and policy quality

Tune rewards and network size, add evaluation-driven self-play/curriculum, and
measure larger rollout/game counts. CUDA is a separate build/runtime milestone;
this first validated lane is CPU only. Deployment parity and repeatable evaluation
should precede expensive large-scale experiments.
