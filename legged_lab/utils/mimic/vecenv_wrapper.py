"""Flatten structured motion observations for RSL-RL's MLP policy."""

from collections.abc import Mapping

import torch
from tensordict import TensorDict, TensorDictBase
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def flatten_observations(observations, num_envs):
    def leaves(value):
        if isinstance(value, (Mapping, TensorDictBase)):
            return [leaf for item in value.values() for leaf in leaves(item)]
        return [value.reshape(num_envs, -1)]

    return TensorDict(
        {name: torch.cat(leaves(value), dim=-1) for name, value in observations.items()},
        batch_size=[num_envs],
    )


class MimicVecEnvWrapper(RslRlVecEnvWrapper):
    def get_observations(self):
        return flatten_observations(super().get_observations(), self.num_envs)

    def reset(self):
        observations, extras = super().reset()
        return flatten_observations(observations, self.num_envs), extras

    def step(self, actions):
        observations, rewards, dones, extras = super().step(actions)
        return flatten_observations(observations, self.num_envs), rewards, dones, extras
