# Environment setup

The project has two development lanes. GigaLearnCPP is the production training
stack. The Python environment is a reference and support stack used to validate
the RLGym/RocketSim behavior and support RLBot tooling.

## Python support environment

Use Python 3.11 in the existing repository environment, `.venv`; it does not need
to be replaced. `requirements.txt` lists the direct reference dependencies, and
`requirements-lock.txt` pins that resolved set. `requirements-training.txt` adds the
CPU Torch wheel and its pinned dependencies. The Windows wheel supplies LibTorch
C++ headers, libraries and CMake configuration. GigaLearnCPP remains the C++ PPO
learner; installing the wheel does not install a Python GigaLearn learner.

From the repository root, refresh the existing environment with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt -r requirements-training.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m rl_bot.smoke
.\.venv\Scripts\python.exe -m rl_bot.check_rocketsim
```

On a new checkout without `.venv`, first install `uv`, run
`uv python install 3.11.16`, and then `uv venv --python 3.11.16 --seed .venv`.
Keep one environment and use its explicit interpreter path in the commands above.

The current checks pass with Python 3.11.16, RLGym 2.0.1 and RocketSim 2.2.1. They
verify the Python reference environment only.

## GigaLearn C++ environment

The source manifest, `dependencies/gigalearn.json`, pins the author-published
GigaLearnCPP commit `86e11c3881a21225a43c36b14af40f0b3c87e7dd` and the Git tree
hashes of its bundled dependencies. These are ordinary directories in the
published snapshot, not gitlink submodules; independent upstream commit IDs are
not supplied. `scripts/bootstrap-training.ps1` fetches the pinned checkout into
ignored `artifacts/gigalearn`, verifies the commit and tree IDs, and applies the
project's reproducible integration patch. Keep third-party source and binaries
local; redistribution terms remain unresolved.

Verified on this Windows machine on 2026-09-16:

- Visual Studio 2022 Build Tools 17.14.41 (installation 17.14.37710.0), with the
  C++ workload, MSVC x64/x86 tools, CMake tools and Windows 11 SDK 26100.
- MSVC compiler 19.44.35229.0 in tools directory 14.44.35207.
- CMake 3.31.6-msvc6; Windows SDK headers/libraries 10.0.26100.0
  (installer bundle 10.1.26100.7705).
- Python 3.11.16, including `Python.h`, `python311.lib` and `python311.dll` in its
  uv-managed base installation.
- Torch 2.14.0+cpu, with `torch.version.cuda` equal to `None`, and available
  LibTorch C++ headers, libraries and CMake configuration in `.venv`.

The toolchain installation completed successfully without a required restart.
Use these commands to check it, build the Release CPU executable and run the
bounded readiness verifier:

```powershell
.\scripts\setup-toolchain.ps1 -CheckOnly
.\scripts\build-training.ps1 -Jobs 1
.\.venv\Scripts\python.exe scripts\verify_training.py
```

If the first check reports missing components on another machine, run
`.\scripts\setup-toolchain.ps1` to install the pinned Build Tools package through
Windows Package Manager. It does not force a restart. The build wrapper discovers
CMake through PATH or Visual Studio Installer, invokes the source bootstrap, and
discovers Python development files from `.venv`'s base interpreter. One compile
job limits memory pressure on the current 16 GB machine.

By default, the build obtains `CMAKE_PREFIX_PATH` from
`torch.utils.cmake_prefix_path`. An external CPU LibTorch 2.14.0 distribution can be
selected with `scripts/build-training.ps1 -TorchPrefix <libtorch-root>`; the
output directory is configurable with `-BuildDirectory`. The default executable
is `build/bin/Release/rl-bot-train.exe`.

Collision meshes remain runtime assets outside version control. The launch and
verification wrappers can use the existing RLGym reference meshes from `.venv`;
see [RocketSim](rocketsim.md) for asset checks and explicit path options. Keep
LibTorch, generated projects, executables, collision meshes, checkpoints and logs
out of version control. Add CUDA only after bounded CPU training saves and resumes
successfully.

## Verification boundary

A successful Python simulator check or toolchain installation alone does not
prove that the C++ learner works. On 2026-09-16 the Release C++ build, both CTest
checks and all nine native acceptance checks passed. This establishes CPU
training readiness for standard-map soccar 1v1s.

`scripts/verify_training.py` is the readiness gate. It checks the native contract
and environment, finite optimizer updates, complete checkpoint saves, resume,
retention, rejection of incompatible/incomplete checkpoints, and a bounded update
with the actual CPU launch configuration. A successful run writes a report to
`artifacts/training/<run-id>/verification.json`. See the
[training workflow](training-and-checkpoints.md) for the verified outcome and
training launch commands.
