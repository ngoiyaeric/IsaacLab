# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


class G1EnergyEnv(ManagerBasedRLEnv):
    """
    Custom environment for the G1 robot with an energy benchmark.
    This intercepts the environment step to track battery state and tokens.
    """

    def __init__(self, cfg, **kwargs):
        # 1. Allocate custom buffers before calling super().__init__()
        # These will be initialized to 1.0 (full battery) and 0.0 (no tokens)
        self.battery_buf = None
        self.tokens_buf = None

        # Super init will call load_managers, so we will initialize the buffers inside load_managers
        super().__init__(cfg, **kwargs)

    def load_managers(self):
        super().load_managers()
        # Initialize the buffers now that the number of environments is known
        self.battery_buf = torch.ones(self.num_envs, device=self.device)
        self.tokens_buf = torch.zeros(self.num_envs, device=self.device)

        # Parameters for energy / tokens
        self.max_battery = self.cfg.battery_capacity
        self.battery_drain_rate = self.cfg.battery_drain_rate
        self.token_earn_rate = self.cfg.token_earn_rate
        self.charge_token_cost = self.cfg.charge_token_cost
        self.charging_station_radius = self.cfg.charging_station_radius

    def step(self, action: torch.Tensor):
        # Process actions
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # Perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()

            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()

            self.scene.update(dt=self.physics_dt)

        # -- UPDATE BUFFERS AND ENERGY/TOKENS --
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Calculate energy drain based on power consumption
        robot = self.scene["robot"]

        # Using simplified energy drain: sum of absolute torques
        # You could also use the actual power formula: |torque * velocity|
        energy_drain = torch.sum(torch.abs(robot.data.applied_torque), dim=1) * self.battery_drain_rate * self.step_dt
        self.battery_buf = torch.clamp(self.battery_buf - energy_drain, min=0.0, max=self.max_battery)

        # Calculate token earning (Job: Tracking velocity)
        # Job quality depends on linear velocity tracking error
        vel_cmd = self.command_manager.get_command("base_velocity")
        current_vel = robot.data.root_lin_vel_b

        vel_error = torch.sum(torch.square(vel_cmd[:, :2] - current_vel[:, :2]), dim=1)

        # Earn tokens if error is low (robot is doing its job well)
        job_quality = torch.exp(-vel_error / 0.5)
        tokens_earned = job_quality * self.token_earn_rate * self.step_dt
        self.tokens_buf += tokens_earned

        # Charging logic
        # Check distance to charging station (origin 0,0)
        dist_to_station = torch.norm(robot.data.root_pos_w[:, :2], dim=1)
        at_station = dist_to_station < self.charging_station_radius
        can_charge = self.tokens_buf >= self.charge_token_cost

        charging_envs = at_station & can_charge

        if charging_envs.any():
            charging_ids = charging_envs.nonzero(as_tuple=False).flatten()

            # Apply charge
            self.battery_buf[charging_ids] = self.max_battery
            self.tokens_buf[charging_ids] -= self.charge_token_cost

            # Trigger custom event for visual / logging if needed
            if "at_charging_station" in self.event_manager.available_modes:
                self.event_manager.apply(mode="at_charging_station", env_ids=charging_ids)

        # -- MANAGERS --
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # Reset environments
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)

            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- COMMANDS & EVENTS --
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def _reset_idx(self, env_ids: torch.Tensor):
        # Sample custom metrics logging before buffer reset
        avg_battery = torch.mean(self.battery_buf[env_ids]).item()
        avg_tokens = torch.mean(self.tokens_buf[env_ids]).item()

        super()._reset_idx(env_ids)

        self.extras["log"]["Metrics/avg_battery"] = avg_battery
        self.extras["log"]["Metrics/avg_tokens"] = avg_tokens

        # Reset custom buffers
        self.battery_buf[env_ids] = self.max_battery
        self.tokens_buf[env_ids] = 0.0
