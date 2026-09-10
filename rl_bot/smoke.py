"""Real simulator reset/step check: python -m rl_bot.smoke."""
import json
import numpy as np
from .environment import build_env, contract

def main():
    env = build_env()
    rng = np.random.default_rng(123)
    try:
        obs = env.reset()
        assert len(obs) == 2
        start = env.state.tick_count
        physics_dtype = str(env.state.ball.position.dtype)
        for _ in range(16):
            actions = {a: np.array([rng.integers(env.action_space(a)[1])], dtype=np.int64) for a in obs}
            obs, rewards, terminated, truncated = env.step(actions)
            assert set(obs) == set(rewards) == set(terminated) == set(truncated)
            assert all(np.isfinite(r) for r in rewards.values())
            assert all(o.dtype == np.float32 and np.isfinite(o).all() for o in obs.values())
            if any(terminated.values()) or any(truncated.values()):
                raise AssertionError("Unexpected early episode end in short smoke run")
        assert env.state.tick_count - start == 128
        print(json.dumps({"contract": contract(), "observations": {a: list(o.shape) for a,o in obs.items()}, "physics_dtype": physics_dtype, "ticks":128}, indent=2))
    finally:
        env.close()

if __name__ == "__main__":
    main()
