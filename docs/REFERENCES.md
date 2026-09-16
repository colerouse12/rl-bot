# Official documentation references

Use this file to locate authoritative information for the project. It is a curated
link map rather than a copy of upstream documentation. Runtime results live in
project notes.

## Version rule

Use installed package source for exact behavior when an upstream project has no tag
matching our pinned release. Use a matching tag or verified commit when one exists.
Consult current `main` or general documentation for concepts, then confirm signatures
against the installed version before implementation.

Python versions are pinned in `requirements.txt`; the full resolved Python set is in
`requirements-lock.txt`; `requirements-training.txt` pins the Windows CPU Torch
wheel and its support dependencies. C++ source identities are recorded in
`dependencies/gigalearn.json`; verified toolchain versions are in the setup note.

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

## GigaLearnCPP training stack

- [Pinned GigaLearnCPP source](https://github.com/ZealanL/GigaLearnCPP-Leak/tree/86e11c3881a21225a43c36b14af40f0b3c87e7dd)
- [Pinned overview and feature list](https://github.com/ZealanL/GigaLearnCPP-Leak/blob/86e11c3881a21225a43c36b14af40f0b3c87e7dd/README.md)
- [Example 1v1 trainer entry point](https://github.com/ZealanL/GigaLearnCPP-Leak/blob/86e11c3881a21225a43c36b14af40f0b3c87e7dd/src/ExampleMain.cpp)
- [Learner configuration](https://github.com/ZealanL/GigaLearnCPP-Leak/blob/86e11c3881a21225a43c36b14af40f0b3c87e7dd/GigaLearnCPP/src/public/GigaLearnCPP/LearnerConfig.h)
- [PPO learner configuration](https://github.com/ZealanL/GigaLearnCPP-Leak/blob/86e11c3881a21225a43c36b14af40f0b3c87e7dd/GigaLearnCPP/src/public/GigaLearnCPP/PPO/PPOLearnerConfig.h)
- [RLGymCPP environment components](https://github.com/ZealanL/GigaLearnCPP-Leak/tree/86e11c3881a21225a43c36b14af40f0b3c87e7dd/GigaLearnCPP/RLGymCPP)

The repository is an archived source snapshot rather than a released package and has
no detected top-level license file. Its dependency folders are ordinary Git trees,
not gitlink submodules despite the included `.gitmodules` file. The manifest records
their tree hashes. The project fetches this snapshot into ignored storage for local
use; redistribution terms remain unresolved. GigaLearn implements PPO internally;
project code uses its learner and checkpoint interfaces rather than RLGym Learn or
RLGym-PPO. Local adaptations are reproducible through `scripts/patch_gigalearn.py`.

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
- [Pinned GGLBot deployment adapter](https://github.com/SubparN0va/GGLBot/tree/d5d21daa7653f842699b590d6bf85597c617e4d8)

The Python-interface repository has no tag matching beta 52, so use installed
`rlbot` source for exact API behavior. The bridge has a matching `v0.3.0` tag.
Its README explains timing and configuration, but its illustrative `Agent` class
is not an installed policy loader; our deployment code must provide that boundary.
GGLBot is the closer reference for loading GigaLearn `.lt` models in an RLBot v5 C++
executable. Its observation builder, action parser and inference layout must still be
matched to this project's final contract, and its redistribution terms must be
checked before copying source.

## C++ toolchain, LibTorch, NumPy 1.26 and Python 3.11

- LibTorch: [C++ installation](https://pytorch.org/cppdocs/installing.html) and
  [platform/download selector](https://pytorch.org/get-started/locally/)
- CMake: [current documentation](https://cmake.org/cmake/help/latest/)
- MSVC: [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- [Visual Studio installer command-line parameters](https://learn.microsoft.com/en-us/visualstudio/install/use-command-line-parameters-to-install-visual-studio?view=vs-2022)
- Windows: [SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate)
  for the supervisor's temporary idle-sleep request (verified 2026-09-16).
- NumPy 1.26: [manual](https://numpy.org/doc/1.26/),
  [copies and views](https://numpy.org/doc/1.26/user/basics.copies.html),
  [broadcasting](https://numpy.org/doc/1.26/user/basics.broadcasting.html),
  [stack](https://numpy.org/doc/1.26/reference/generated/numpy.stack.html), and
  [finite checks](https://numpy.org/doc/1.26/reference/generated/numpy.isfinite.html)
- Python 3.11: [virtual environments](https://docs.python.org/3.11/library/venv.html)
- Environment bootstrapping: [Python downloads](https://www.python.org/downloads/)
  and [uv Python/environment management](https://docs.astral.sh/uv/guides/install-python/)

LibTorch releases, ABI details and CMake configuration can move independently of the
archived GigaLearn source. Record the exact compatible LibTorch archive and compiler
version after the first successful build. Python NumPy remains part of the reference
lane rather than the production model runtime.

## Adding a reference

Add only the official page needed for an active architecture or implementation
question. Include the applicable package version or verified commit when available.
Record project conclusions in the owning project note, not in this link map.
