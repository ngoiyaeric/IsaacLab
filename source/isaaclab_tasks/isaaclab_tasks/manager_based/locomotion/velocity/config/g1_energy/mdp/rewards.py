# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


def battery_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the robot when its battery gets too low."""
    # Scale from 0 (full) to -1 (empty) linearly
    # You could make it non-linear e.g., only penalize if below 20%
    return -(1.0 - (env.battery_buf / env.max_battery))


def empty_battery_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Heavily penalize when battery hits exactly 0."""
    return torch.where(
        env.battery_buf <= 0.01,
        torch.tensor(-10.0, device=env.device),
        torch.tensor(0.0, device=env.device),
    )
