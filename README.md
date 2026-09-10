# Rocket League 1v1 bot

This repository is in the architecture and contract-validation phase for a 1v1 bot.
Training will use RLGym Learn; deployment will use RLBot v5 through the RLGym bridge.
Use **Python 3.11 in `.venv`**.

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -m pip check
python -m rl_bot.smoke
python -m rl_bot.check_rocketsim
```

The dependency check and both simulator checks pass. These checks cover dependency
compatibility, native physics, 1v1 resets, observations, actions, goals and timeout
behavior. They intentionally do not claim that a training workflow exists yet.

Project documentation starts at the [documentation index](docs/README.md).
The [development plan](docs/project/development-plan.md) defines the architecture
milestones, while the [training workflow](docs/project/training-and-checkpoints.md)
records the intended operational contract as that architecture is implemented.
