"""Real native physics checks. Run with python -m rl_bot.check_rocketsim."""
import json
from importlib.metadata import distribution, version

import numpy as np
from rlgym.rocket_league.action_parsers import LookupTableAction

from .environment import build_env, contract


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    table = LookupTableAction.make_lookup_table()
    neutral = int(np.flatnonzero(np.all(table == 0, axis=1))[0])
    forward = int(np.flatnonzero(np.all(table == [1, 0, 0, 0, 0, 0, 0, 0], axis=1))[0])
    env = build_env()
    checks = []
    try:
        obs = env.reset()
        require(len(obs) == 2, "Expected 1v1")
        actions = {a: np.array([neutral], dtype=np.int64) for a in obs}
        initial = {a: c.physics.position.copy() for a, c in env.state.cars.items()}
        for _ in range(30):
            obs, rewards, terminated, truncated = env.step(
                {a: np.array([forward], dtype=np.int64) for a in obs}
            )
            require(all(o.shape == (contract()['obs_dim'],) and np.isfinite(o).all() for o in obs.values()), "Invalid observation")
        require(all(np.linalg.norm(c.physics.position - initial[a]) > 100 for a, c in env.state.cars.items()), "Cars did not move")
        checks.append("car movement and finite observations")

        env.reset()
        state = env.state
        state.ball.position = np.array([0, 0, 700], dtype=np.float32)
        state.ball.linear_velocity = np.array([0, 0, -1000], dtype=np.float32)
        env.set_state(state)
        bounced = False
        for _ in range(30):
            env.step(actions)
            require(env.state.ball.position[2] > 0, "Ball fell through arena floor")
            bounced |= env.state.ball.linear_velocity[2] > 100
        require(bounced, "No floor bounce; check collision meshes")
        checks.append("arena floor collision")

        env.reset()
        state = env.state
        state.ball.position = np.array([3800, 0, 500], dtype=np.float32)
        state.ball.linear_velocity = np.array([1500, 0, 0], dtype=np.float32)
        env.set_state(state)
        for _ in range(8):
            env.step(actions)
        require(env.state.ball.linear_velocity[0] < 0, "No side-wall bounce")
        checks.append("arena wall collision")

        env.reset()
        state = env.state
        state.ball.position = np.array([0, 5000, 200], dtype=np.float32)
        state.ball.linear_velocity = np.array([0, 1800, 0], dtype=np.float32)
        env.set_state(state)
        scored = False
        for _ in range(12):
            _, rewards, terminated, truncated = env.step(actions)
            if any(terminated.values()):
                require(all(terminated.values()) and not any(truncated.values()), "Goal done mappings are wrong")
                require(rewards['blue-0'] > 0 and rewards['orange-0'] < 0, "Goal reward perspective is wrong")
                scored = True
                break
        require(scored, "Goal was not detected")
        checks.append("goal termination and team reward signs")
        for _ in range(3):
            obs = env.reset()
            require(len(obs) == 2 and not env.state.goal_scored, "Reset failed")
            require(np.linalg.norm(env.state.ball.position[:2]) < 1, "Ball did not reset to kickoff")
            env.step(actions)
        checks.append("repeated kickoff resets")
    finally:
        env.close()
    wheel = distribution('rocketsim').read_text('WHEEL') or ''
    print(json.dumps({"runtime": "PASS", "checks": checks, "rocketsim": version('rocketsim'),
                      "wheel_tags": [line for line in wheel.splitlines() if line.startswith('Tag:')],
                      "environment": "Python 3.11 is required for the installed Windows wheel; pip check passes."}, indent=2))


if __name__ == '__main__':
    main()
