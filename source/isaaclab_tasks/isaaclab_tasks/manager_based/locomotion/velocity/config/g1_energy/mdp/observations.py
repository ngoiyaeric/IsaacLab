# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


def battery_level(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the current normalized battery level of the robot."""
    # (num_envs, 1)
    return (env.battery_buf / env.max_battery).unsqueeze(-1)


def token_count(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the current token count of the robot."""
    # (num_envs, 1)
    return env.tokens_buf.unsqueeze(-1)
