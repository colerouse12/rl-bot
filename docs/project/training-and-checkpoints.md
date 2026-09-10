# Training workflow and checkpoints

Status reviewed 2026-09-10. This is the operating guide for the 1v1 bot;
[development plan](development-plan.md) tracks the work needed to complete it.

## Current architecture phase

Python 3.11 in `.venv` runs the shared RLGym environment and native physics checks.
The training CLI, policy exporter and RLBot launcher are intentionally future
milestones. This guide defines their contracts before implementation. Commands below
are runnable unless explicitly labelled as a proposed interface.

The selected learner is RLGym Learn 2.0.0 with rlgym-learn-algos 0.4.2. RLGym-PPO
uses different configuration and checkpoint APIs. The [setup note](setup.md) owns
dependencies, [RocketSim](rocketsim.md) owns simulator checks, and
[official references](../REFERENCES.md) owns external links.

## 1. Preflight

Open PowerShell in the repository root and use the interpreter explicitly:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m rl_bot.smoke
.\.venv\Scripts\python.exe -m rl_bot.check_rocketsim
```

Expected: Python 3.11, no broken requirements, and passing simulator checks. Any
dependency error, nonfinite observation, missing mesh, or native crash must be
investigated before training. Do not edit package metadata or use `--no-deps` to
turn a failing compatibility check green.

The simulator uses meshes already available in the installed RLGym package. Do not
copy proprietary game assets into the repository.

## 2. Understand the environment contract

[environment.py](../../rl_bot/environment.py) is the shared implementation:

- `build_env()` creates one blue and one orange car using kickoff resets.
- `BotObs` produces a fresh, finite float32 array of shape `[92]` for each agent.
  Dimensions are reported by the builder and checked at runtime.
- Position scales use side-wall, back-net and ceiling dimensions; velocity scales
  use car maximum linear/angular speed. Remaining feature order follows the pinned
  `DefaultObs`. Orange observations use its inverted perspective.
- `make_actions()` uses the pinned 90-row lookup table with eight physics ticks per
  selected action. Each agent supplies an int64 array of shape `[1]`.
- `RocketSimEngine(rlbot_delay=True)` keeps the one-tick action delay. Preserve this
  setting in deployment.
- Rewards combine goal reward with weight 10 and touch reward with weight 0.1.
  This is a minimal baseline, not a proven reward design.
- Goals terminate episodes. No touch for 30 simulated seconds or an episode lasting
  300 simulated seconds truncates them. Truncation must retain the appropriate
  value bootstrap in trajectory processing; do not collapse both done flags.

`reset()` returns the per-agent observation mapping. `step()` returns observation,
reward, termination and truncation mappings. The learner must handle both agents
and reset completed environments. Eight physics ticks, one environment decision,
and two agent transitions are different units: report each counter explicitly.

`contract()` currently records dimensions, dtype, team size, action repetition,
delay and lookup-table hash. It does not yet encode every feature/scaling decision.
Before policy export, extend its versioning or metadata to detect those changes.

## 3. First optimizer update — implementation pending

Start with one environment worker and CPU execution. The initial run should be
short enough to inspect, but collect enough experience for at least one complete
PPO update. Model dimensions must come from discovered spaces, not copied example
constants. Use importable worker factories and a guarded main entry point on Windows.

Configure these exact installed-package fields explicitly:

- `ProcessConfigModel.n_proc`: worker count; begin with 1.
- `BaseConfigModel.serde_types`: string agent IDs, float32 observations, int64
  action arrays, Python float rewards and tuple space descriptions.
- `PPOAgentControllerConfigModel.timesteps_per_iteration`: rollout threshold.
- `PPOLearnerConfigModel.batch_size`, `n_minibatches`, `n_epochs`: update sizing.
- `ExperienceBufferConfigModel.max_size`: enough capacity for the intended batches.
- `PPOLearnerConfigModel.device`, `dtype`, optimizer groups and learning rates.
- Controller `save_every_ts`, `n_checkpoints_to_keep`, and unique run directory.

Do not adopt the upstream 32-worker/GPU/W&B example as the local default. Use local
metrics and no external logging. Set numerical-library thread limits before imports
if workers oversubscribe the CPU; measure throughput before increasing concurrency.
Choose CPU first. GPU support and its installation path require a separate verified
configuration after the baseline works.

Acceptance requires finite losses, gradients and parameters; evidence of an actual
optimizer step; and a nonzero parameter change. Upstream `PPOData` exposes entropy,
KL divergence, critic loss, clip fraction, actor/critic update magnitudes and
`cumulative_model_updates`. That counter increments by consumed batches in this
release, so it is not an exact count of every optimizer object's `step()` call.

Test observation batches `[1,92]` and `[B,92]`, action indices, log probabilities,
values, returns and advantages. Explicitly prevent `[B]` versus `[B,1]` broadcasting.
The actual policy-output and learner-boundary shapes must be recorded from the
running implementation before declaring this stage complete.

## 4. Run, observe, and stop — implementation pending

For each run record a unique ID, start time, source revision and dirty state, package
versions, seed, device, worker/thread counts, resolved config and contract metadata.
Record agent transitions, wall time, throughput, completed episodes, goals, touches,
timeouts, return statistics and PPO diagnostics. Seeded runs are useful comparisons;
parallel simulation is not automatically bitwise reproducible.

Do not treat rising shaped reward as evidence of stronger play. Compare changes with
fixed evaluation seeds and opponents. Stop and investigate NaNs, zero update
magnitudes, repeated worker failures or a stopped transition counter.

The installed coordinator supports keyboard `p` (pause), `c` (checkpoint), and `q`
(checkpoint and quit), polled every 50 loop iterations. This is source-inspected,
not tested through a project launcher. Its `start()` catches exceptions, attempts
a save and cleans up; a printed traceback may not produce a failing process exit.
The launcher must surface failures and verify worker shutdown.

A normal timestep-limit exit does not explicitly save in the coordinator's run
loop. Add and test a final-save path. Define whether the project's step limit means
additional transitions this invocation or lifetime transitions, including resume.
Do not assume the coordinator and restored PPO controller counters are synchronized.

## 5. Save and resume — source inspected, runtime unverified

The installed `PPOAgentController.save_checkpoint()` creates timestamp-named
checkpoint folders. A full checkpoint can include:

- Learner `actor_critic.pt`, `optimizers.pt` and `misc.json` with the model-update
  counter.
- Controller `ppo_agent.json` with iteration and transition counters.
- Experience-buffer and trajectory-processor state, subject to configuration.
- Optional metrics state and pickled mid-iteration trajectories/shared information.

Automatic saves are checked against `save_every_ts`; the save counter is advanced
after a learning iteration. The setting is not a precise wall-clock save interval.
Retention defaults to five checkpoints and deletes older numeric directories.
Use a dedicated per-run checkpoint directory containing only learner checkpoints;
keep exported policies, notes and selected permanent snapshots elsewhere.

The loader restores model/optimizer state and maps tensors to the configured
device. Some missing auxiliary files cause warnings and fresh counters or buffers.
Require a checkpoint-completeness check so a partial restore is not called a full
resume. Full snapshots include pickle data: load only trusted local checkpoints.
For the project policy loader, make trusted-file handling and tensor-only loading
explicit rather than relying on changing PyTorch defaults.

Resume acceptance: save after an update, exit, launch a fresh process, load the
checkpoint, compare policy outputs on fixed observations, verify optimizer and
counter restoration, then perform another update. Test CPU loading separately.
Do not promise exact continuation of simulator state or random streams unless
those states are explicitly captured and restored.

## 6. Evaluate, export, deploy — implementation pending

Evaluation disables gradient tracking, puts the policy in evaluation mode and
records deterministic versus sampled action selection. Use fixed episode budgets,
seeds and opponent identities; report wins/losses/draws alongside goal difference,
timeouts and sample count. Begin with random/neutral baselines and a frozen policy
snapshot. These sanity checks do not establish a competitive rank.

A policy-only artifact must contain actor weights, architecture, full observation
and action contract, package versions and training/checkpoint identity. It is not
a resumable training checkpoint. Reject incompatible metadata before inference.
Test batch size one and exported-policy action equivalence on CPU.

RLBot deployment must import the same observation/action factories and match their
timing. The bridge README's illustrative `Agent` is not an installed policy loader.
Implement it for this learner, then test player identity, both team perspectives,
kickoffs, non-playing phases and reset behavior in an offline local match.

## Proposed command interface — not available yet

Reserve a small interface for the implementation: `python -m rl_bot.train` with
`--steps`, `--workers`, `--device`, `--run-dir`, and optional `--resume`; separate
`rl_bot.evaluate` and `rl_bot.export_policy` modules for evaluation and export.
These module names are a proposal, not runnable instructions. Replace this section
with exact tested invocations and observed results when the commands exist.

## Sources and maintenance

Project-authored notes were inspected 2026-09-10 against installed RLGym Learn
2.0.0 and rlgym-learn-algos 0.4.2. Exact upstream links are owned by
[official references](../REFERENCES.md). Relevant installed symbols are
`LearningCoordinator.start`, `LearningCoordinator._run`,
`PPOAgentController.save_checkpoint`, `PPOLearner.save_checkpoint`, and the PPO
experience-buffer implementation.

Recheck these symbols when dependency versions change. Update commands, evidence
and limitations together; documented behavior and runtime verification remain separate.
