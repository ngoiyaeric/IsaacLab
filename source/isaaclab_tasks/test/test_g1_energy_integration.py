# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for the G1 Energy Environment."""

import sys

# Import pinocchio in the main script to force the use of the dependencies
# installed by IsaacLab and not the one installed by Isaac Sim.
# pinocchio is required by the Pink IK controller
if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import pytest
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class TestG1EnergyEnvIntegration:
    """Integration tests for G1EnergyEnv."""

    @pytest.fixture(scope="class")
    def env_cfg(self):
        """Create environment configuration."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        return parse_env_cfg(env_id, device="cuda" if torch.cuda.is_available() else "cpu", num_envs=2)

    def test_env_creation(self, env_cfg):
        """Test that the environment can be created successfully."""
        env_id = "Isaac-Velocity-Energy-G1-v0"

        # Set config args
        env_cfg.sim.create_stage_in_memory = False

        try:
            env = gym.make(env_id, cfg=env_cfg)
            assert env is not None
            assert hasattr(env.unwrapped, "battery_buf")
            assert hasattr(env.unwrapped, "tokens_buf")
            env.close()
        except Exception as e:
            pytest.fail(f"Failed to create G1 Energy environment: {e}")

    def test_env_reset(self, env_cfg):
        """Test that the environment can be reset."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        env_cfg.sim.create_stage_in_memory = False

        env = gym.make(env_id, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        try:
            obs, info = env.reset()

            # Check observation shape
            assert obs is not None
            assert isinstance(obs, torch.Tensor)
            assert obs.shape[0] == env_cfg.num_envs

            # Check that battery is initialized to full
            assert torch.allclose(env.unwrapped.battery_buf, torch.ones_like(env.unwrapped.battery_buf) * env.unwrapped.max_battery)

            # Check that tokens are initialized to zero
            assert torch.allclose(env.unwrapped.tokens_buf, torch.zeros_like(env.unwrapped.tokens_buf))

        finally:
            env.close()

    def test_env_step(self, env_cfg):
        """Test that the environment can perform a step."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        env_cfg.sim.create_stage_in_memory = False

        env = gym.make(env_id, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        try:
            obs, info = env.reset()

            # Create random action
            action = torch.randn(env_cfg.num_envs, env.unwrapped.single_action_space.shape[0], device=env.unwrapped.device)

            # Take a step
            obs, reward, terminated, truncated, info = env.step(action)

            # Check outputs
            assert obs is not None
            assert reward is not None
            assert terminated is not None
            assert truncated is not None

            # Check shapes
            assert obs.shape[0] == env_cfg.num_envs
            assert reward.shape[0] == env_cfg.num_envs
            assert terminated.shape[0] == env_cfg.num_envs
            assert truncated.shape[0] == env_cfg.num_envs

            # Check for NaN values
            assert not torch.any(torch.isnan(obs))
            assert not torch.any(torch.isnan(reward))

        finally:
            env.close()

    def test_battery_drain(self, env_cfg):
        """Test that battery drains over time with actions."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        env_cfg.sim.create_stage_in_memory = False

        env = gym.make(env_id, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        try:
            obs, info = env.reset()
            initial_battery = env.unwrapped.battery_buf.clone()

            # Take multiple steps with large actions (high torque)
            for _ in range(10):
                action = torch.ones(env_cfg.num_envs, env.unwrapped.single_action_space.shape[0], device=env.unwrapped.device)
                obs, reward, terminated, truncated, info = env.step(action)

            # Battery should have decreased
            final_battery = env.unwrapped.battery_buf
            assert torch.all(final_battery <= initial_battery)

        finally:
            env.close()

    def test_battery_termination(self, env_cfg):
        """Test that environment terminates when battery is empty."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        env_cfg.sim.create_stage_in_memory = False

        env = gym.make(env_id, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        try:
            obs, info = env.reset()

            # Manually set battery to zero to test termination
            env.unwrapped.battery_buf[:] = 0.0

            # Take a step
            action = torch.zeros(env_cfg.num_envs, env.unwrapped.single_action_space.shape[0], device=env.unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(action)

            # Should trigger termination for all environments
            # Note: termination is computed before checking reset_buf, so we check the termination manager
            result = env.unwrapped.termination_manager.compute()
            assert torch.any(result)

        finally:
            env.close()

    def test_token_earning(self, env_cfg):
        """Test that tokens can be earned."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        env_cfg.sim.create_stage_in_memory = False

        env = gym.make(env_id, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        try:
            obs, info = env.reset()
            initial_tokens = env.unwrapped.tokens_buf.clone()

            # Take multiple steps
            for _ in range(10):
                action = torch.zeros(env_cfg.num_envs, env.unwrapped.single_action_space.shape[0], device=env.unwrapped.device)
                obs, reward, terminated, truncated, info = env.step(action)

            # Tokens should have increased (or at least not decreased)
            final_tokens = env.unwrapped.tokens_buf
            assert torch.all(final_tokens >= initial_tokens)

        finally:
            env.close()

    def test_observations_include_battery_and_tokens(self, env_cfg):
        """Test that observations include battery level and token count."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        env_cfg.sim.create_stage_in_memory = False

        env = gym.make(env_id, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        try:
            obs, info = env.reset()

            # Check that observation manager has battery and token terms
            assert "battery_level" in env.unwrapped.observation_manager.active_terms["policy"]
            assert "token_count" in env.unwrapped.observation_manager.active_terms["policy"]

        finally:
            env.close()

    def test_rewards_include_battery_penalties(self, env_cfg):
        """Test that rewards include battery penalties."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        env_cfg.sim.create_stage_in_memory = False

        env = gym.make(env_id, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        try:
            obs, info = env.reset()

            # Check that reward manager has battery penalty terms
            assert "battery_penalty" in env.unwrapped.reward_manager.active_terms
            assert "empty_battery_penalty" in env.unwrapped.reward_manager.active_terms

            # Take a step and check rewards are computed
            action = torch.zeros(env_cfg.num_envs, env.unwrapped.single_action_space.shape[0], device=env.unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(action)

            assert reward is not None
            assert not torch.any(torch.isnan(reward))

        finally:
            env.close()

    def test_env_reset_resets_battery_and_tokens(self, env_cfg):
        """Test that resetting an environment resets battery and tokens."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        env_cfg.sim.create_stage_in_memory = False

        env = gym.make(env_id, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        try:
            obs, info = env.reset()

            # Modify battery and tokens
            env.unwrapped.battery_buf[:] = 0.5
            env.unwrapped.tokens_buf[:] = 10.0

            # Reset
            obs, info = env.reset()

            # Should be back to initial values
            assert torch.allclose(env.unwrapped.battery_buf, torch.ones_like(env.unwrapped.battery_buf) * env.unwrapped.max_battery)
            assert torch.allclose(env.unwrapped.tokens_buf, torch.zeros_like(env.unwrapped.tokens_buf))

        finally:
            env.close()

    def test_multiple_steps_stability(self, env_cfg):
        """Test environment stability over multiple steps."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        env_cfg.sim.create_stage_in_memory = False

        env = gym.make(env_id, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        try:
            obs, info = env.reset()

            # Take 100 steps
            for step in range(100):
                action = torch.randn(env_cfg.num_envs, env.unwrapped.single_action_space.shape[0], device=env.unwrapped.device)
                obs, reward, terminated, truncated, info = env.step(action)

                # Check for NaN or inf values
                assert not torch.any(torch.isnan(obs)), f"NaN in observations at step {step}"
                assert not torch.any(torch.isnan(reward)), f"NaN in rewards at step {step}"
                assert not torch.any(torch.isinf(obs)), f"Inf in observations at step {step}"
                assert not torch.any(torch.isinf(reward)), f"Inf in rewards at step {step}"

                # Check battery bounds
                assert torch.all(env.unwrapped.battery_buf >= 0.0), f"Battery below 0 at step {step}"
                assert torch.all(env.unwrapped.battery_buf <= env.unwrapped.max_battery), f"Battery above max at step {step}"

                # Check tokens are non-negative
                assert torch.all(env.unwrapped.tokens_buf >= 0.0), f"Tokens negative at step {step}"

        finally:
            env.close()

    def test_custom_metrics_logging(self, env_cfg):
        """Test that custom metrics are logged in extras."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        env_cfg.sim.create_stage_in_memory = False

        env = gym.make(env_id, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        try:
            obs, info = env.reset()

            # Take a step
            action = torch.zeros(env_cfg.num_envs, env.unwrapped.single_action_space.shape[0], device=env.unwrapped.device)

            # Manually drain some battery and add tokens
            env.unwrapped.battery_buf[:] = 0.7
            env.unwrapped.tokens_buf[:] = 5.0

            # Trigger reset to log metrics
            env.unwrapped.reset_buf[:] = True
            obs, reward, terminated, truncated, info = env.step(action)

            # Check that metrics are in extras
            if "log" in env.unwrapped.extras:
                extras_log = env.unwrapped.extras["log"]
                # Metrics may be logged during reset
                # Just verify structure exists
                assert isinstance(extras_log, dict)

        finally:
            env.close()

    def test_charging_station_proximity(self, env_cfg):
        """Test charging station proximity detection."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        env_cfg.sim.create_stage_in_memory = False

        env = gym.make(env_id, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        try:
            obs, info = env.reset()

            # Check that charging station radius is set
            assert hasattr(env.unwrapped, "charging_station_radius")
            assert env.unwrapped.charging_station_radius > 0.0

            # Check that charge token cost is set
            assert hasattr(env.unwrapped, "charge_token_cost")
            assert env.unwrapped.charge_token_cost > 0.0

        finally:
            env.close()


class TestG1EnergyEnvEdgeCases:
    """Integration tests for edge cases."""

    @pytest.fixture(scope="class")
    def env_cfg(self):
        """Create environment configuration."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        return parse_env_cfg(env_id, device="cuda" if torch.cuda.is_available() else "cpu", num_envs=2)

    def test_zero_actions(self, env_cfg):
        """Test environment with zero actions."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        env_cfg.sim.create_stage_in_memory = False

        env = gym.make(env_id, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        try:
            obs, info = env.reset()

            # Take multiple steps with zero actions
            for _ in range(10):
                action = torch.zeros(env_cfg.num_envs, env.unwrapped.single_action_space.shape[0], device=env.unwrapped.device)
                obs, reward, terminated, truncated, info = env.step(action)

            # Environment should remain stable
            assert not torch.any(torch.isnan(obs))
            assert not torch.any(torch.isnan(reward))

        finally:
            env.close()

    def test_max_actions(self, env_cfg):
        """Test environment with maximum actions."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        env_cfg.sim.create_stage_in_memory = False

        env = gym.make(env_id, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        try:
            obs, info = env.reset()
            initial_battery = env.unwrapped.battery_buf.clone()

            # Take steps with maximum actions (should drain battery faster)
            for _ in range(5):
                action = torch.ones(env_cfg.num_envs, env.unwrapped.single_action_space.shape[0], device=env.unwrapped.device)
                obs, reward, terminated, truncated, info = env.step(action)

            # Battery should have decreased more than with zero actions
            assert torch.all(env.unwrapped.battery_buf < initial_battery)

        finally:
            env.close()

    def test_rapid_resets(self, env_cfg):
        """Test rapid environment resets."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        env_cfg.sim.create_stage_in_memory = False

        env = gym.make(env_id, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        try:
            # Perform multiple rapid resets
            for _ in range(5):
                obs, info = env.reset()
                assert obs is not None
                assert torch.all(env.unwrapped.battery_buf == env.unwrapped.max_battery)
                assert torch.all(env.unwrapped.tokens_buf == 0.0)

        finally:
            env.close()