# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


def battery_empty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Indicates whether an episode should terminate when the environment's battery level is depleted.
    
    Parameters:
    	env (ManagerBasedRLEnv): Environment whose `battery_buf` tensor is checked for depletion.
    
    Returns:
    	torch.Tensor: Boolean tensor with `True` where `battery_buf` is less than or equal to 0.0, `False` otherwise.
    """
    return env.battery_buf <= 0.0
