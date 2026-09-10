# Official documentation references

Use this file to locate authoritative information for the project. It is a curated
link map rather than a copy of upstream documentation. Runtime results live in
project notes.

## Version rule

Use installed package source for exact behavior when an upstream project has no tag
matching our pinned release. Use a matching tag or verified commit when one exists.
Consult current `main` or general documentation for concepts, then confirm signatures
against the installed version before implementation.

Direct versions are pinned in `requirements.txt`; the full resolved set is in
`requirements-lock.txt`.

## RLGym 2.0.1 and the Rocket League environment API

- [RLGym documentation home](https://rlgym.org/)
- [Quick start and environment construction](https://rlgym.org/Getting%20Started/quickstart/)
- Configuration objects: [actions](https://rlgym.org/Rocket%20League/Configuration%20Objects/action_parsers/),
  [observations](https://rlgym.org/Rocket%20League/Configuration%20Objects/observation_builders/),
  [rewards](https://rlgym.org/Rocket%20League/Configuration%20Objects/reward_functions/),
  [done conditions](https://rlgym.org/Rocket%20League/Configuration%20Objects/done_conditions/),
  and [state mutators](https://rlgym.org/Rocket%20League/Configuration%20Objects/state_mutators/)
- [RLGym core source](https://github.com/RLGym/rlgym)
- [Rocket League package source](https://github.com/lucas-emery/rocket-league-gym)
- [RLGym 2.0.1 package](https://pypi.org/project/rlgym/2.0.1/)
- [Rocket League 2.0.1 package](https://pypi.org/project/rlgym-rocket-league/2.0.1/)

Neither source repository exposes a verified `v2.0.1` tag. For exact signatures,
inspect `.venv/Lib/site-packages/rlgym/api` and
`.venv/Lib/site-packages/rlgym/rocket_league`. Key symbols include `RLGym`,
`DefaultObs`, `LookupTableAction`, `RepeatAction`, `GameState`, `Car`,
`PhysicsObject`, `GoalCondition`, `TimeoutCondition`, `KickoffMutator`, and
`RocketSimEngine`.

## RLGym Learn PPO stack

- [RLGym Learn v2.0.0 source](https://github.com/JPK314/rlgym-learn/tree/250379331d21946afaed384dd0fadc857f3df6b6)
- [Version-matched quick-start example](https://github.com/JPK314/rlgym-learn/blob/250379331d21946afaed384dd0fadc857f3df6b6/quick_start_guide.py)
- [RLGym Learn Algorithms v0.4.2 source](https://github.com/JPK314/rlgym-learn-algos/tree/466b8e30f911914602a5cb3ba6053eabc53ab0e3)
- [RLGym Learn 2.0.0 package](https://pypi.org/project/rlgym-learn/2.0.0/)
- [Algorithms 0.4.2 package](https://pypi.org/project/rlgym-learn-algos/0.4.2/)

For exact worker, serialization, PPO and checkpoint behavior, inspect installed
`rlgym_learn` and `rlgym_learn_algos/ppo`. Start with `LearningCoordinator`,
`PPOAgentControllerConfigModel`, `PPOLearnerConfigModel`,
`ExperienceBufferConfigModel`, `PPOLearner.save_checkpoint`, and
`PPOAgentController.save_checkpoint`. RLGym-PPO is a separate trainer; its examples
are not interchangeable with this stack.

## RocketSim 2.2.1

- [Python binding v2.2.1](https://github.com/mtheall/RocketSim/tree/v2.2.1)
- [Python package](https://pypi.org/project/rocketsim/2.2.1/)
- [Original RocketSim project](https://github.com/ZealanL/RocketSim)
- [Original engine documentation](https://zealanl.github.io/RocketSimDocs/)

Use the Python binding for installation and Python API details. The original C++
project explains the simulator model and arena assets. Project-specific prerequisites
and passing checks are in `docs/project/rocketsim.md`.

## RLBot v5 and the RLGym bridge

- [RLBot v5 wiki](https://wiki.rlbot.org/v5/)
- [RLBotServer lifecycle and control loop](https://wiki.rlbot.org/v5/framework/sockets-specification/)
- [Bot and match configuration](https://wiki.rlbot.org/v5/botmaking/config-files/)
- [RLBot Python interface](https://github.com/RLBot/python-interface)
- [RLBot 2.0.0b52 package](https://pypi.org/project/rlbot/2.0.0b52/)
- [RLGym RLBot bridge v0.3.0](https://github.com/RLGym/rlgym-rlbot/tree/v0.3.0)
- [Bridge 0.3.0 package](https://pypi.org/project/rlgym-rlbot/0.3.0/)

The Python-interface repository has no tag matching beta 52, so use installed
`rlbot` source for exact API behavior. The bridge has a matching `v0.3.0` tag.
Its README explains timing and configuration, but its illustrative `Agent` class
is not an installed policy loader; our deployment code must provide that boundary.

## PyTorch 2.14, NumPy 1.26 and Python 3.11

- PyTorch: [tensor basics](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html),
  [autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html),
  [saving and loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html),
  [inference mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html),
  and [Categorical distributions](https://docs.pytorch.org/docs/stable/distributions.html#categorical)
- NumPy 1.26: [manual](https://numpy.org/doc/1.26/),
  [copies and views](https://numpy.org/doc/1.26/user/basics.copies.html),
  [broadcasting](https://numpy.org/doc/1.26/user/basics.broadcasting.html),
  [stack](https://numpy.org/doc/1.26/reference/generated/numpy.stack.html), and
  [finite checks](https://numpy.org/doc/1.26/reference/generated/numpy.isfinite.html)
- Python 3.11: [virtual environments](https://docs.python.org/3.11/library/venv.html),
  [multiprocessing](https://docs.python.org/3.11/library/multiprocessing.html), and
  [spawn guidance](https://docs.python.org/3.11/library/multiprocessing.html#the-spawn-and-forkserver-start-methods)
- Environment bootstrapping: [Python downloads](https://www.python.org/downloads/)
  and [uv Python/environment management](https://docs.astral.sh/uv/guides/install-python/)

PyTorch’s stable web documentation may move ahead of our pin. Confirm checkpoint
loading defaults and signatures against installed PyTorch 2.14.0 before implementing
the policy and checkpoint boundary.

## Adding a reference

Add only the official page needed for an active architecture or implementation
question. Include the applicable package version or verified commit when available.
Record project conclusions in the owning project note, not in this link map.
