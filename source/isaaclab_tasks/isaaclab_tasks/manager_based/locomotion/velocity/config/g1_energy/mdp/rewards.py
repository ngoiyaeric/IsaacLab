# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


def battery_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Compute a linear penalty proportional to remaining battery.
    
    Parameters:
        env (ManagerBasedRLEnv): Environment providing `battery_buf` and `max_battery`.
    
    Returns:
        torch.Tensor: Penalty computed as -(1.0 - (env.battery_buf / env.max_battery)); values are 0 when battery is full and approach -1 as battery approaches empty.
    """
    # Scale from 0 (full) to -1 (empty) linearly
    # You could make it non-linear e.g., only penalize if below 20%
    return -(1.0 - (env.battery_buf / env.max_battery))


def empty_battery_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Apply a heavy penalty when the environment's battery is effectively empty.
    
    Parameters:
        env (ManagerBasedRLEnv): Environment exposing `battery_buf` (current battery level),
            `device` (tensor device), and `max_battery` (not used here).
    
    Returns:
        torch.Tensor: A scalar tensor on `env.device` with value `-10.0` if `env.battery_buf <= 0.01`,
        `0.0` otherwise.
    """
    return torch.where(
        env.battery_buf <= 0.01,
        torch.tensor(-10.0, device=env.device),
        torch.tensor(0.0, device=env.device),
    )
