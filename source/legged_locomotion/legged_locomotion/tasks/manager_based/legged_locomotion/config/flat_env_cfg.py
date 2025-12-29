# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        import inspect, os


        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None
        # terminations
        self.terminations.base_height.params["sensor_cfg"] = None  # world-frame height is fine
        self.terminations.terrain_out_of_bounds = None  # no need


class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.command_levels = None
