# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .env import G1EnergyEnv
from .env_cfg import G1EnergyEnvCfg

gym.register(
    id="Isaac-Velocity-Energy-G1-v0",
    entry_point="isaaclab_tasks.manager_based.locomotion.velocity.config.g1_energy:G1EnergyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1EnergyEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)
