"""Shared training/deployment observation and action configuration."""
import hashlib
import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.obs_builders import DefaultObs

ACTION_REPEAT = 8

class BotObs(DefaultObs):
    def __init__(self):
        super().__init__(
            zero_padding=1,
            pos_coef=np.array([1 / cv.SIDE_WALL_X, 1 / cv.BACK_NET_Y, 1 / cv.CEILING_Z]),
            lin_vel_coef=1 / cv.CAR_MAX_SPEED,
            ang_vel_coef=1 / cv.CAR_MAX_ANG_VEL,
        )

    def build_obs(self, agents, state, shared_info):
        observations = super().build_obs(agents, state, shared_info)
        for agent, obs in observations.items():
            obs = np.array(obs, dtype=np.float32, copy=True)
            if obs.shape != (self.get_obs_space(agent)[1],) or not np.isfinite(obs).all():
                raise ValueError("Invalid 1v1 observation shape or nonfinite features")
            observations[agent] = obs
        return observations

def make_actions():
    return RepeatAction(LookupTableAction(), repeats=ACTION_REPEAT)

def contract():
    table = LookupTableAction.make_lookup_table().astype("<i8")
    return {
        "schema": 1, "obs_builder": "rl_bot.environment.BotObs",
        "obs_dim": BotObs().get_obs_space("blue-0")[1],
        "obs_dtype": "float32", "action_count": len(table),
        "action_repeat": ACTION_REPEAT,
        "action_table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
        "rlbot_delay": True, "team_size": 1,
    }

def build_env():
    from rlgym.rocket_league.done_conditions import AnyCondition, GoalCondition, NoTouchTimeoutCondition, TimeoutCondition
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence
    return RLGym(
        state_mutator=MutatorSequence(FixedTeamSizeMutator(1, 1), KickoffMutator()),
        obs_builder=BotObs(), action_parser=make_actions(),
        reward_fn=CombinedReward((GoalReward(), 10.0), (TouchReward(), 0.1)),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(NoTouchTimeoutCondition(30), TimeoutCondition(300)),
        transition_engine=RocketSimEngine(rlbot_delay=True),
    )
