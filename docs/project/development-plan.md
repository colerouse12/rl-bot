# Development plan: train and deploy a 1v1 bot

Reviewed 2026-09-10. The first deliverable is a reproducible local loop:
simulate a 1v1 match, update a policy, save it, resume training, evaluate it, and run
the exported policy in an offline RLBot match. Competitive rank is not the initial
acceptance criterion. The [training workflow](training-and-checkpoints.md) explains
operation and upstream behavior; this file owns milestone status and completion gates.

## Current baseline

Implemented and tested: shared 1v1 environment, 92-feature float32 observations,
90 discrete actions repeated for eight ticks, goals/touches baseline rewards,
kickoff resets and native physics checks. See [environment.py](../../rl_bot/environment.py),
[smoke.py](../../rl_bot/smoke.py), and [check_rocketsim.py](../../rl_bot/check_rocketsim.py).

Selected stack: Python 3.11 in `.venv`, RLGym Learn 2.0.0 plus algorithms 0.4.2,
RLGym 2.0.1, RocketSim 2.2.1 and the RLBot v5 bridge. Package pins are in
[requirements.txt](../../requirements.txt). No change of interpreter or learner is
planned. Dependency metadata and simulator runtime checks pass.

Current design scope: training launcher, model factories, policy loader/exporter,
checkpoint lifecycle, evaluation harness and bot/match launch configuration. Their
interfaces are being defined before implementation. External sources are curated in
[official references](../REFERENCES.md).

## Milestone 1 — one verified PPO update (next)

Build an importable worker factory around `build_env`, space-driven actor/critic
factories, explicit serializers and a guarded Windows entry point. Start with one
CPU worker, a small bounded rollout and local metrics. Validate config sizes and
worker failure reporting before allocating resources.

Completion checks:

- [ ] Real simulator observations travel through worker serialization to the model.
- [ ] Both agents contribute trajectories with correct termination/truncation handling.
- [ ] At least one optimizer update changes weights; losses and parameters stay finite.
- [ ] Single-observation and multi-observation batch shape tests pass without broadcasting.
- [ ] The step limit stops the run and all owned workers exit.
- [ ] Exact command, config and measured result are recorded in the workflow guide.

Preserve the verified Python 3.11 dependency baseline while implementing the first
bounded runtime experiment. Re-run dependency and simulator checks after package changes.

## Milestone 2 — dependable stop, save and resume

Use a unique output directory for each run. Capture resolved config, versions and
contract metadata. Add final-save behavior, explicit process exit status and
checkpoint completeness validation. Define additional-versus-lifetime step semantics.
Protect permanent snapshots from automatic retention.

Completion checks:

- [ ] Normal exit and a graceful requested stop each produce a loadable checkpoint.
- [ ] A fresh process restores model, optimizer and counters and performs another update.
- [ ] Fixed-observation outputs match before save and after load.
- [ ] CPU checkpoint loading works; incompatible or incomplete checkpoints fail clearly.
- [ ] Retention only removes eligible checkpoints inside the designated run directory.
- [ ] Failure-path tests leave no worker processes running or false success message.

Simulator state and RNG restoration are separate work if exact continuation is
required; restoring learner state alone must not be described as exact replay.

## Milestone 3 — policy export and repeatable evaluation

Implement a small CPU inference loader and policy-only export, with complete
feature/scaling/perspective and action-table metadata. Build fixed-seed evaluation
against simple baselines and a frozen snapshot. Compare policies under identical
conditions and keep training and evaluation trajectories separate.

Completion checks:

- [ ] Exported and in-training actors agree on a fixed observation suite.
- [ ] Batch size one, blue/orange perspectives and invalid artifact metadata are tested.
- [ ] Evaluation records policy hash, opponent, seeds, episode count and outcome metrics.
- [ ] No gradients or optimizer updates occur in evaluation.
- [ ] A saved policy can be evaluated from a fresh process with a documented command.

## Milestone 4 — RLBot offline deployment

Add bot/match configuration and a policy-loading adapter using the pinned bridge.
Share `BotObs` and `make_actions`; preserve the simulator delay/action repetition
contract. Implement policy loading explicitly rather than copying the placeholder
Agent in the bridge example.

Completion checks:

- [ ] Bot configuration parses and loads the expected policy on CPU.
- [ ] A local 1v1 match launches and both blue/orange assignments work.
- [ ] Kickoffs, goals, resets and non-playing phases behave correctly.
- [ ] Actions are valid and inference/tick timing is measured on the target machine.
- [ ] Launch prerequisites and actual tested steps are recorded separately from assumptions.

Rocket League and RLBot Core are external prerequisites. Package imports or simulator
tests cannot substitute for a real match launch.

## Milestone 5 — controlled training improvement

Only after the full loop works, measure sustained throughput and resource usage,
increase workers gradually, and evaluate GPU usefulness. Change one experimental
variable at a time: reward weights, model size or PPO configuration. Preserve a
frozen baseline and promote policies using evaluation results, not training return
alone. Set a run budget and stopping criterion before each long run.

Completion checks:

- [ ] A bounded longer run remains stable and its checkpoints resume successfully.
- [ ] Resource settings and throughput are measured and reproducible enough to compare.
- [ ] Each experiment records its hypothesis, change, budget and evaluation outcome.

## Risks and decisions to track

- **Packaging:** keep Python 3.11 and the locked dependency set unless a full
  compatibility check justifies a change. Re-run native checks before scaling workers.
- **Checkpoint semantics:** the upstream coordinator can swallow exceptions; counters
  and final saves need project-level tests. A printed save message is insufficient.
- **Contract drift:** changes to feature order/scaling, perspective or action ordering
  can invalidate a policy without changing tensor dimensions. Version these explicitly.
- **Reward exploitation:** touch reward can increase while match play gets worse.
  Evaluate outcomes against fixed opponents before expanding reward shaping.
- **Runtime/deployment differences:** the wrapper approximates simulator timing and
  state. Measure live behavior before drawing conclusions from simulation alone.

CPU is the initial device. Model size, rollout/batch sizes and later GPU settings
remain implementation decisions to measure, not established project facts.

## Development discipline and handoff

Keep changes focused and preserve existing local work. Use the local-first routing
in [AGENTS.md](../../AGENTS.md). Add tests for meaningful boundaries and failure
modes; update operating instructions in the same change as new commands. Keep
checkpoints, environments, assets and large logs out of Git. Commit/push only when
requested.

For each completed milestone record its command, environment, result, artifact
location and outstanding limitations. Mark a checkbox complete only with evidence.
The immediate next task is Milestone 1; milestones are ordered by dependencies,
not calendar estimates.

Deferred: replay ingestion, behavior cloning, replay-derived resets, advanced
curricula, distributed training, opponent leagues and team modes beyond 1v1.
Do not install dependencies for those features during this initial loop.
