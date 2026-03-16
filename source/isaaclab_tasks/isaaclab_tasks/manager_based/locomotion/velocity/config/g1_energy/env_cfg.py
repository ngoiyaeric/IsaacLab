# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)

import source.isaaclab_tasks.isaaclab_tasks.manager_based.locomotion.velocity.config.g1_energy.mdp as custom_mdp


@configclass
class G1EnergyEnvCfg(G1FlatEnvCfg):
    """Configuration for the G1 Energy Environment."""

    # Energy / Token configurations
    battery_capacity: float = 1.0
    battery_drain_rate: float = 0.005  # Rate of battery drain based on torque
    token_earn_rate: float = 0.1  # Tokens earned per second of good tracking
    charge_token_cost: float = 1.0  # Cost in tokens to fully charge
    charging_station_radius: float = 1.0  # Radius to trigger charging at origin

    def __post_init__(self):
        """
        Finalize post-initialization for the energy environment configuration by applying energy-specific command limits, episode length, terminations, rewards, observations, and event tracking.
        
        This method adjusts the base movement command ranges, extends the episode duration to accommodate charging behavior, registers a battery-empty termination, adds battery-related reward terms (including an empty-battery penalty), exposes battery level and token count as policy observations, and enables tracking of the "at_charging_station" event.
        """
        super().__post_init__()

        # Overwrite the base environment commands
        # Allow the robot to track X, Y and Yaw
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Extend episode length to allow for longer charging cycles
        self.episode_length_s = 60.0

        # Terminations
        self.terminations.battery_empty = DoneTerm(func=custom_mdp.battery_empty, time_out=True)

        # Rewards
        self.rewards.battery_penalty = RewTerm(
            func=custom_mdp.battery_penalty, weight=0.1
        )  # Note: returns negative inside
        self.rewards.empty_battery_penalty = RewTerm(func=custom_mdp.empty_battery_penalty, weight=1.0)

        # Observations
        # Add the custom observation terms to the policy observation space
        self.observations.policy.battery_level = ObsTerm(func=custom_mdp.battery_level)
        self.observations.policy.token_count = ObsTerm(func=custom_mdp.token_count)

        # Ensure event mode is tracked
        self.events.at_charging_station = EventTerm(func=lambda env, env_ids: None, mode="at_charging_station")
