# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


def battery_level(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Get the robot's battery level normalized to its maximum capacity.
    
    Parameters:
        env (ManagerBasedRLEnv): Environment containing `battery_buf` and `max_battery`.
    
    Returns:
        torch.Tensor: Tensor of shape (num_envs, 1) with values in [0, 1] representing each environment's battery level divided by `max_battery`.
    """
    # (num_envs, 1)
    return (env.battery_buf / env.max_battery).unsqueeze(-1)


def token_count(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Provide the current token count per environment as a single-column tensor.
    
    Returns:
        torch.Tensor: Tensor of shape (num_envs, 1) containing the token count for each environment.
    """
    # (num_envs, 1)
    return env.tokens_buf.unsqueeze(-1)
