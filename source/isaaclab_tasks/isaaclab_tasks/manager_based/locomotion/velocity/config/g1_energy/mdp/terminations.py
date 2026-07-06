# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


def battery_empty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode if the battery level reaches 0."""
    return env.battery_buf <= 0.0
