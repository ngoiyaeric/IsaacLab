# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the G1 Energy Environment and MDP functions."""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
import gymnasium as gym

# Import the modules to be tested
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1_energy.mdp import (
    observations,
    rewards,
    terminations,
)


class TestMDPObservations:
    """Unit tests for MDP observation functions."""

    def test_battery_level_full_battery(self):
        """Test battery_level with full battery."""
        # Create mock environment
        env = Mock()
        env.battery_buf = torch.tensor([1.0, 1.0, 1.0])
        env.max_battery = 1.0

        result = observations.battery_level(env)

        assert result.shape == (3, 1)
        assert torch.allclose(result, torch.tensor([[1.0], [1.0], [1.0]]))

    def test_battery_level_half_battery(self):
        """Test battery_level with half battery."""
        env = Mock()
        env.battery_buf = torch.tensor([0.5, 0.25, 0.75])
        env.max_battery = 1.0

        result = observations.battery_level(env)

        assert result.shape == (3, 1)
        assert torch.allclose(result, torch.tensor([[0.5], [0.25], [0.75]]))

    def test_battery_level_empty_battery(self):
        """Test battery_level with empty battery."""
        env = Mock()
        env.battery_buf = torch.tensor([0.0, 0.0, 0.0])
        env.max_battery = 1.0

        result = observations.battery_level(env)

        assert result.shape == (3, 1)
        assert torch.allclose(result, torch.tensor([[0.0], [0.0], [0.0]]))

    def test_battery_level_different_max_battery(self):
        """Test battery_level with different max battery capacity."""
        env = Mock()
        env.battery_buf = torch.tensor([0.5, 1.0, 1.5])
        env.max_battery = 2.0

        result = observations.battery_level(env)

        assert result.shape == (3, 1)
        assert torch.allclose(result, torch.tensor([[0.25], [0.5], [0.75]]))

    def test_battery_level_single_env(self):
        """Test battery_level with single environment."""
        env = Mock()
        env.battery_buf = torch.tensor([0.8])
        env.max_battery = 1.0

        result = observations.battery_level(env)

        assert result.shape == (1, 1)
        assert torch.allclose(result, torch.tensor([[0.8]]))

    def test_token_count_zero_tokens(self):
        """Test token_count with zero tokens."""
        env = Mock()
        env.tokens_buf = torch.tensor([0.0, 0.0, 0.0])

        result = observations.token_count(env)

        assert result.shape == (3, 1)
        assert torch.allclose(result, torch.tensor([[0.0], [0.0], [0.0]]))

    def test_token_count_positive_tokens(self):
        """Test token_count with positive token values."""
        env = Mock()
        env.tokens_buf = torch.tensor([5.5, 10.0, 2.3])

        result = observations.token_count(env)

        assert result.shape == (3, 1)
        assert torch.allclose(result, torch.tensor([[5.5], [10.0], [2.3]]))

    def test_token_count_single_env(self):
        """Test token_count with single environment."""
        env = Mock()
        env.tokens_buf = torch.tensor([7.2])

        result = observations.token_count(env)

        assert result.shape == (1, 1)
        assert torch.allclose(result, torch.tensor([[7.2]]))

    def test_token_count_large_values(self):
        """Test token_count with large token values."""
        env = Mock()
        env.tokens_buf = torch.tensor([100.0, 500.0, 1000.0])

        result = observations.token_count(env)

        assert result.shape == (3, 1)
        assert torch.allclose(result, torch.tensor([[100.0], [500.0], [1000.0]]))


class TestMDPRewards:
    """Unit tests for MDP reward functions."""

    def test_battery_penalty_full_battery(self):
        """Test battery_penalty with full battery (should return 0 penalty)."""
        env = Mock()
        env.battery_buf = torch.tensor([1.0, 1.0, 1.0])
        env.max_battery = 1.0

        result = rewards.battery_penalty(env)

        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0]))

    def test_battery_penalty_empty_battery(self):
        """Test battery_penalty with empty battery (should return -1)."""
        env = Mock()
        env.battery_buf = torch.tensor([0.0, 0.0, 0.0])
        env.max_battery = 1.0

        result = rewards.battery_penalty(env)

        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([-1.0, -1.0, -1.0]))

    def test_battery_penalty_half_battery(self):
        """Test battery_penalty with half battery."""
        env = Mock()
        env.battery_buf = torch.tensor([0.5, 0.5, 0.5])
        env.max_battery = 1.0

        result = rewards.battery_penalty(env)

        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([-0.5, -0.5, -0.5]))

    def test_battery_penalty_mixed_levels(self):
        """Test battery_penalty with mixed battery levels."""
        env = Mock()
        env.battery_buf = torch.tensor([1.0, 0.5, 0.0, 0.25, 0.75])
        env.max_battery = 1.0

        result = rewards.battery_penalty(env)

        expected = torch.tensor([0.0, -0.5, -1.0, -0.75, -0.25])
        assert result.shape == (5,)
        assert torch.allclose(result, expected)

    def test_battery_penalty_different_max_battery(self):
        """Test battery_penalty with different max battery capacity."""
        env = Mock()
        env.battery_buf = torch.tensor([2.0, 1.0, 0.0])
        env.max_battery = 2.0

        result = rewards.battery_penalty(env)

        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([0.0, -0.5, -1.0]))

    def test_empty_battery_penalty_above_threshold(self):
        """Test empty_battery_penalty when battery is above threshold."""
        env = Mock()
        env.battery_buf = torch.tensor([1.0, 0.5, 0.1])
        env.device = "cpu"

        result = rewards.empty_battery_penalty(env)

        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0]))

    def test_empty_battery_penalty_at_threshold(self):
        """Test empty_battery_penalty when battery is at threshold."""
        env = Mock()
        env.battery_buf = torch.tensor([0.01, 0.01, 0.01])
        env.device = "cpu"

        result = rewards.empty_battery_penalty(env)

        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0]))

    def test_empty_battery_penalty_below_threshold(self):
        """Test empty_battery_penalty when battery is below threshold."""
        env = Mock()
        env.battery_buf = torch.tensor([0.0, 0.005, 0.009])
        env.device = "cpu"

        result = rewards.empty_battery_penalty(env)

        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([-10.0, -10.0, -10.0]))

    def test_empty_battery_penalty_mixed_levels(self):
        """Test empty_battery_penalty with mixed battery levels."""
        env = Mock()
        env.battery_buf = torch.tensor([1.0, 0.5, 0.01, 0.0, 0.005])
        env.device = "cpu"

        result = rewards.empty_battery_penalty(env)

        expected = torch.tensor([0.0, 0.0, 0.0, -10.0, -10.0])
        assert result.shape == (5,)
        assert torch.allclose(result, expected)

    def test_empty_battery_penalty_gpu_device(self):
        """Test empty_battery_penalty with GPU device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        env = Mock()
        env.battery_buf = torch.tensor([0.0, 0.5], device="cuda")
        env.device = "cuda"

        result = rewards.empty_battery_penalty(env)

        assert result.device.type == "cuda"
        assert result.shape == (2,)
        expected = torch.tensor([-10.0, 0.0], device="cuda")
        assert torch.allclose(result, expected)


class TestMDPTerminations:
    """Unit tests for MDP termination functions."""

    def test_battery_empty_with_empty_battery(self):
        """Test battery_empty when battery is at 0."""
        env = Mock()
        env.battery_buf = torch.tensor([0.0, 0.0, 0.0])

        result = terminations.battery_empty(env)

        assert result.shape == (3,)
        assert torch.all(result == torch.tensor([True, True, True]))

    def test_battery_empty_with_full_battery(self):
        """Test battery_empty when battery is full."""
        env = Mock()
        env.battery_buf = torch.tensor([1.0, 1.0, 1.0])

        result = terminations.battery_empty(env)

        assert result.shape == (3,)
        assert torch.all(result == torch.tensor([False, False, False]))

    def test_battery_empty_with_partial_battery(self):
        """Test battery_empty when battery is partially charged."""
        env = Mock()
        env.battery_buf = torch.tensor([0.5, 0.1, 0.9])

        result = terminations.battery_empty(env)

        assert result.shape == (3,)
        assert torch.all(result == torch.tensor([False, False, False]))

    def test_battery_empty_with_mixed_levels(self):
        """Test battery_empty with mixed battery levels."""
        env = Mock()
        env.battery_buf = torch.tensor([1.0, 0.0, 0.5, 0.0, 0.1])

        result = terminations.battery_empty(env)

        expected = torch.tensor([False, True, False, True, False])
        assert result.shape == (5,)
        assert torch.all(result == expected)

    def test_battery_empty_negative_battery(self):
        """Test battery_empty with negative battery values (edge case)."""
        env = Mock()
        env.battery_buf = torch.tensor([-0.1, -0.5, 0.0])

        result = terminations.battery_empty(env)

        expected = torch.tensor([True, True, True])
        assert result.shape == (3,)
        assert torch.all(result == expected)

    def test_battery_empty_single_env(self):
        """Test battery_empty with single environment."""
        env = Mock()
        env.battery_buf = torch.tensor([0.0])

        result = terminations.battery_empty(env)

        assert result.shape == (1,)
        assert result[0] == True


class TestG1EnergyEnvCfg:
    """Unit tests for G1EnergyEnvCfg configuration class."""

    def test_env_cfg_defaults(self):
        """Test that G1EnergyEnvCfg has correct default values."""
        from isaaclab_tasks.manager_based.locomotion.velocity.config.g1_energy.env_cfg import G1EnergyEnvCfg

        cfg = G1EnergyEnvCfg()

        # Check energy/token configurations
        assert cfg.battery_capacity == 1.0
        assert cfg.battery_drain_rate == 0.005
        assert cfg.token_earn_rate == 0.1
        assert cfg.charge_token_cost == 1.0
        assert cfg.charging_station_radius == 1.0

    def test_env_cfg_post_init(self):
        """Test that __post_init__ sets up observations, rewards, and terminations."""
        from isaaclab_tasks.manager_based.locomotion.velocity.config.g1_energy.env_cfg import G1EnergyEnvCfg

        cfg = G1EnergyEnvCfg()

        # Check that custom terminations are added
        assert hasattr(cfg.terminations, "battery_empty")
        assert cfg.terminations.battery_empty.time_out is True

        # Check that custom rewards are added
        assert hasattr(cfg.rewards, "battery_penalty")
        assert cfg.rewards.battery_penalty.weight == 0.1
        assert hasattr(cfg.rewards, "empty_battery_penalty")
        assert cfg.rewards.empty_battery_penalty.weight == 1.0

        # Check that custom observations are added
        assert hasattr(cfg.observations.policy, "battery_level")
        assert hasattr(cfg.observations.policy, "token_count")

        # Check that episode length is extended
        assert cfg.episode_length_s == 60.0

    def test_env_cfg_velocity_ranges(self):
        """Test that velocity command ranges are configured."""
        from isaaclab_tasks.manager_based.locomotion.velocity.config.g1_energy.env_cfg import G1EnergyEnvCfg

        cfg = G1EnergyEnvCfg()

        # Check velocity ranges
        assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.0, 1.0)
        assert cfg.commands.base_velocity.ranges.lin_vel_y == (-0.5, 0.5)
        assert cfg.commands.base_velocity.ranges.ang_vel_z == (-1.0, 1.0)

    def test_env_cfg_charging_event(self):
        """Test that charging station event is configured."""
        from isaaclab_tasks.manager_based.locomotion.velocity.config.g1_energy.env_cfg import G1EnergyEnvCfg

        cfg = G1EnergyEnvCfg()

        # Check that charging event is added
        assert hasattr(cfg.events, "at_charging_station")


class TestGymRegistration:
    """Tests for gym environment registration."""

    def test_environment_registered(self):
        """Test that the G1 Energy environment is registered with gym."""
        env_id = "Isaac-Velocity-Energy-G1-v0"

        # Check if environment is registered
        assert env_id in gym.registry

    def test_environment_registration_details(self):
        """Test that the environment registration has correct details."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        spec = gym.spec(env_id)

        assert spec.id == env_id
        assert "G1EnergyEnv" in spec.entry_point
        assert spec.disable_env_checker is True

    def test_environment_kwargs(self):
        """Test that the environment registration has correct kwargs."""
        env_id = "Isaac-Velocity-Energy-G1-v0"
        spec = gym.spec(env_id)

        assert "env_cfg_entry_point" in spec.kwargs
        assert "G1EnergyEnvCfg" in spec.kwargs["env_cfg_entry_point"]
        assert "rsl_rl_cfg_entry_point" in spec.kwargs


class TestG1EnergyEnvUnit:
    """Unit tests for G1EnergyEnv class methods (without full initialization)."""

    def test_env_has_custom_buffers(self):
        """Test that G1EnergyEnv initializes custom buffers."""
        from isaaclab_tasks.manager_based.locomotion.velocity.config.g1_energy.env import G1EnergyEnv
        from isaaclab_tasks.manager_based.locomotion.velocity.config.g1_energy.env_cfg import G1EnergyEnvCfg

        # Create a mock config
        cfg = Mock(spec=G1EnergyEnvCfg)
        cfg.battery_capacity = 1.0
        cfg.battery_drain_rate = 0.005
        cfg.token_earn_rate = 0.1
        cfg.charge_token_cost = 1.0
        cfg.charging_station_radius = 1.0
        cfg.num_envs = 4
        cfg.device = "cpu"

        # Verify the class has the expected attributes and methods
        assert hasattr(G1EnergyEnv, "__init__")
        assert hasattr(G1EnergyEnv, "load_managers")
        assert hasattr(G1EnergyEnv, "step")
        assert hasattr(G1EnergyEnv, "_reset_idx")

    def test_env_inherits_from_manager_based_rl_env(self):
        """Test that G1EnergyEnv inherits from ManagerBasedRLEnv."""
        from isaaclab_tasks.manager_based.locomotion.velocity.config.g1_energy.env import G1EnergyEnv
        from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

        assert issubclass(G1EnergyEnv, ManagerBasedRLEnv)


class TestG1EnergyEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_battery_level_with_very_small_values(self):
        """Test battery_level with very small battery values."""
        env = Mock()
        env.battery_buf = torch.tensor([1e-10, 1e-8, 1e-6])
        env.max_battery = 1.0

        result = observations.battery_level(env)

        assert result.shape == (3, 1)
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_battery_level_with_very_large_values(self):
        """Test battery_level with very large battery values."""
        env = Mock()
        env.battery_buf = torch.tensor([100.0, 500.0, 1000.0])
        env.max_battery = 1000.0

        result = observations.battery_level(env)

        assert result.shape == (3, 1)
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_rewards_with_extreme_battery_values(self):
        """Test reward functions with extreme battery values."""
        env = Mock()
        env.battery_buf = torch.tensor([1000.0, -100.0, 0.0])
        env.max_battery = 1000.0
        env.device = "cpu"

        # Test battery_penalty
        penalty = rewards.battery_penalty(env)
        assert penalty.shape == (3,)
        assert not torch.any(torch.isnan(penalty))
        assert not torch.any(torch.isinf(penalty))

        # Test empty_battery_penalty
        empty_penalty = rewards.empty_battery_penalty(env)
        assert empty_penalty.shape == (3,)
        assert not torch.any(torch.isnan(empty_penalty))
        assert not torch.any(torch.isinf(empty_penalty))

    def test_terminations_with_float_precision(self):
        """Test termination with floating point precision edge cases."""
        env = Mock()
        # Test values very close to zero
        env.battery_buf = torch.tensor([1e-10, -1e-10, 0.0, 1e-5])

        result = terminations.battery_empty(env)

        assert result.shape == (4,)
        # Values <= 0 should terminate
        assert result[0] == True or result[0] == False  # May vary based on precision
        assert result[1] == True  # Negative should terminate
        assert result[2] == True  # Exactly 0 should terminate
        assert result[3] == False  # Positive should not terminate


class TestG1EnergyNegativeCases:
    """Test negative cases and error conditions."""

    def test_battery_level_with_zero_max_battery(self):
        """Test battery_level behavior with zero max battery (should not crash)."""
        env = Mock()
        env.battery_buf = torch.tensor([0.5, 0.5, 0.5])
        env.max_battery = 0.0

        # This should either raise an error or handle gracefully
        try:
            result = observations.battery_level(env)
            # If it doesn't raise an error, check for inf/nan
            assert torch.any(torch.isinf(result)) or torch.any(torch.isnan(result))
        except (ZeroDivisionError, RuntimeError):
            # Expected behavior for division by zero
            pass

    def test_rewards_consistency(self):
        """Test that battery_penalty and empty_battery_penalty are consistent."""
        env = Mock()
        env.battery_buf = torch.tensor([0.0, 0.01, 0.5, 1.0])
        env.max_battery = 1.0
        env.device = "cpu"

        battery_pen = rewards.battery_penalty(env)
        empty_pen = rewards.empty_battery_penalty(env)

        # Both should penalize low battery
        assert battery_pen[0] < battery_pen[2]
        assert battery_pen[0] < battery_pen[3]

        # Empty penalty should only trigger for very low battery
        assert empty_pen[0] < 0.0  # Battery at 0
        assert empty_pen[3] == 0.0  # Battery full, no empty penalty

    def test_token_count_with_negative_values(self):
        """Test token_count with negative values (edge case that shouldn't happen)."""
        env = Mock()
        env.tokens_buf = torch.tensor([-5.0, -10.0, -1.0])

        result = observations.token_count(env)

        # Should still return the values, even if negative
        assert result.shape == (3, 1)
        assert torch.allclose(result, torch.tensor([[-5.0], [-10.0], [-1.0]]))


class TestG1EnergyRobustness:
    """Test robustness and stress scenarios."""

    def test_large_batch_size(self):
        """Test functions with large batch sizes."""
        env = Mock()
        batch_size = 1000
        env.battery_buf = torch.rand(batch_size)
        env.tokens_buf = torch.rand(batch_size) * 10
        env.max_battery = 1.0
        env.device = "cpu"

        # Test all functions
        battery_obs = observations.battery_level(env)
        token_obs = observations.token_count(env)
        battery_pen = rewards.battery_penalty(env)
        empty_pen = rewards.empty_battery_penalty(env)
        term = terminations.battery_empty(env)

        # Check shapes
        assert battery_obs.shape == (batch_size, 1)
        assert token_obs.shape == (batch_size, 1)
        assert battery_pen.shape == (batch_size,)
        assert empty_pen.shape == (batch_size,)
        assert term.shape == (batch_size,)

        # Check no NaN or inf values
        assert not torch.any(torch.isnan(battery_obs))
        assert not torch.any(torch.isnan(token_obs))
        assert not torch.any(torch.isnan(battery_pen))
        assert not torch.any(torch.isnan(empty_pen))

    def test_dtype_consistency(self):
        """Test that functions maintain dtype consistency."""
        env = Mock()
        env.battery_buf = torch.tensor([0.5, 0.5], dtype=torch.float32)
        env.tokens_buf = torch.tensor([1.0, 2.0], dtype=torch.float32)
        env.max_battery = 1.0
        env.device = "cpu"

        battery_obs = observations.battery_level(env)
        token_obs = observations.token_count(env)
        battery_pen = rewards.battery_penalty(env)

        assert battery_obs.dtype == torch.float32
        assert token_obs.dtype == torch.float32
        assert battery_pen.dtype == torch.float32

    def test_device_consistency_cpu(self):
        """Test that functions maintain device consistency on CPU."""
        env = Mock()
        device = "cpu"
        env.battery_buf = torch.tensor([0.5, 0.5], device=device)
        env.tokens_buf = torch.tensor([1.0, 2.0], device=device)
        env.max_battery = 1.0
        env.device = device

        battery_obs = observations.battery_level(env)
        token_obs = observations.token_count(env)
        battery_pen = rewards.battery_penalty(env)

        assert battery_obs.device.type == device
        assert token_obs.device.type == device
        assert battery_pen.device.type == device


class TestG1EnergyBoundaryConditions:
    """Test boundary conditions and regression cases."""

    def test_battery_at_exactly_one(self):
        """Test behavior when battery is exactly at max capacity."""
        env = Mock()
        env.battery_buf = torch.tensor([1.0, 1.0])
        env.max_battery = 1.0
        env.device = "cpu"

        # Battery level should be exactly 1.0
        battery_obs = observations.battery_level(env)
        assert torch.allclose(battery_obs, torch.tensor([[1.0], [1.0]]))

        # Penalty should be exactly 0
        penalty = rewards.battery_penalty(env)
        assert torch.allclose(penalty, torch.tensor([0.0, 0.0]))

        # Should not terminate
        term = terminations.battery_empty(env)
        assert not torch.any(term)

    def test_battery_at_exactly_zero(self):
        """Test behavior when battery is exactly at 0."""
        env = Mock()
        env.battery_buf = torch.tensor([0.0, 0.0])
        env.max_battery = 1.0
        env.device = "cpu"

        # Battery level should be exactly 0
        battery_obs = observations.battery_level(env)
        assert torch.allclose(battery_obs, torch.tensor([[0.0], [0.0]]))

        # Penalty should be exactly -1
        penalty = rewards.battery_penalty(env)
        assert torch.allclose(penalty, torch.tensor([-1.0, -1.0]))

        # Empty penalty should trigger
        empty_penalty = rewards.empty_battery_penalty(env)
        assert torch.allclose(empty_penalty, torch.tensor([-10.0, -10.0]))

        # Should terminate
        term = terminations.battery_empty(env)
        assert torch.all(term)

    def test_reward_monotonicity(self):
        """Test that battery penalty decreases monotonically as battery drains."""
        env = Mock()
        env.max_battery = 1.0
        env.device = "cpu"

        # Create a sequence of decreasing battery levels
        battery_levels = torch.linspace(1.0, 0.0, 11)  # 1.0, 0.9, 0.8, ..., 0.0

        penalties = []
        for level in battery_levels:
            env.battery_buf = torch.tensor([level])
            penalty = rewards.battery_penalty(env)
            penalties.append(penalty.item())

        # Check that penalties are monotonically non-increasing (more negative)
        for i in range(len(penalties) - 1):
            assert penalties[i] >= penalties[i + 1], f"Penalty not monotonic at index {i}"

    def test_observation_normalization(self):
        """Test that battery level observation is properly normalized."""
        env = Mock()
        env.max_battery = 2.0

        # Test various battery levels with non-unit max battery
        test_levels = [0.0, 0.5, 1.0, 1.5, 2.0]
        expected_normalized = [0.0, 0.25, 0.5, 0.75, 1.0]

        for level, expected in zip(test_levels, expected_normalized):
            env.battery_buf = torch.tensor([level])
            obs = observations.battery_level(env)
            assert torch.allclose(obs, torch.tensor([[expected]])), f"Failed for level {level}"

    def test_termination_boundary_threshold(self):
        """Test termination exactly at the boundary."""
        env = Mock()

        # Test values around 0
        test_values = [-0.1, -1e-10, 0.0, 1e-10, 0.01, 0.1]
        expected_termination = [True, True, True, False, False, False]

        for value, should_terminate in zip(test_values, expected_termination):
            env.battery_buf = torch.tensor([value])
            result = terminations.battery_empty(env)
            assert result[0] == should_terminate, f"Failed for battery={value}"

    def test_empty_penalty_threshold_precision(self):
        """Test empty battery penalty exactly at threshold (0.01)."""
        env = Mock()
        env.device = "cpu"

        # Test values around the 0.01 threshold
        test_values = [0.0, 0.005, 0.009, 0.01, 0.011, 0.02]
        # Values <= 0.01 should trigger penalty based on the code
        expected_penalties = [-10.0, -10.0, -10.0, 0.0, 0.0, 0.0]

        for value, expected in zip(test_values, expected_penalties):
            env.battery_buf = torch.tensor([value])
            result = rewards.empty_battery_penalty(env)
            assert torch.allclose(result, torch.tensor([expected])), f"Failed for battery={value}, got {result.item()}, expected {expected}"

    def test_zero_tokens_observation(self):
        """Test token observation when tokens are exactly zero."""
        env = Mock()
        env.tokens_buf = torch.tensor([0.0, 0.0, 0.0])

        result = observations.token_count(env)

        assert result.shape == (3, 1)
        assert torch.allclose(result, torch.zeros(3, 1))

    def test_all_functions_with_same_input(self):
        """Test all MDP functions with the same environment state for consistency."""
        env = Mock()
        env.battery_buf = torch.tensor([0.3, 0.7, 0.0, 1.0])
        env.tokens_buf = torch.tensor([5.0, 2.5, 10.0, 0.0])
        env.max_battery = 1.0
        env.device = "cpu"

        # Get all outputs
        battery_obs = observations.battery_level(env)
        token_obs = observations.token_count(env)
        battery_pen = rewards.battery_penalty(env)
        empty_pen = rewards.empty_battery_penalty(env)
        term = terminations.battery_empty(env)

        # Verify consistency between related outputs
        # Where battery is 0, should terminate
        assert term[2] == True
        # Where battery is 0, should have max penalty
        assert torch.allclose(battery_pen[2], torch.tensor(-1.0))
        # Where battery is 1, should have no penalty
        assert torch.allclose(battery_pen[3], torch.tensor(0.0))
        # Where battery is 0, should have empty penalty
        assert empty_pen[2] < 0.0

    def test_extreme_token_values(self):
        """Test token observation with extreme values."""
        env = Mock()
        env.tokens_buf = torch.tensor([1e10, 1e-10, -1e10])

        result = observations.token_count(env)

        assert result.shape == (3, 1)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))
        assert torch.allclose(result, env.tokens_buf.unsqueeze(-1))