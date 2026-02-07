# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from legged_locomotion.tasks.manager_based.legged_locomotion.leggedlocomotion_velocity_env_cfg import LocomotionVelocityRoughEnvCfg


# Pre-defined configs
# from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


from go2_location_control.assets.unitree import UNITREE_GO2_CFG



@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        self.actions.joint_pos.scale = 0.25

        # event
        # 事件（Event）相关调整
        # self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


        # rewards
        self.rewards.velocity_tracking_xy.weight = 10.0
        self.rewards.velocity_tracking_xy_fine_grained.weight = 25.0
        self.rewards.velocity_tracking_yaw.weight = 10.0
        self.rewards.velocity_tracking_yaw_fine_grained.weight = 25.0

        # penalties
        self.rewards.lin_vel_z_l2.weight = -5.0
        self.rewards.ang_vel_xy_l2.weight = -1e-1
        self.rewards.dof_torques_l2.weight = -5e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -5e-1
        self.rewards.undesired_contacts.weight = -1e3
        self.rewards.flat_orientation_l2.weight = -50.0
        self.rewards.dof_pos_limits.weight = -5e2
        self.rewards.base_height_l2.weight = -1e2
        self.rewards.body_lin_acc_l2.weight = -5e-4

        # style
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_slide.weight = -2.0
        self.rewards.speed_limit.weight = 0 #1.0
        self.rewards.cheetah.weight = 3.0
        self.rewards.velocity_driven_gait.weight = 0 #2.0
        self.stand_with_all_feet = 2.0

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # curriculums
        # self.curriculum.terrain_levels = None


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        # 调用父类后初始化以继承上面的所有设置
        super().__post_init__()


        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        # 减少生成的子地形数量以节省内存与加速渲染（play 模式）
        if self.scene.terrain.terrain_generator is not None:
            n = max(1, int(self.scene.num_envs ** 0.5))
            self.scene.terrain.terrain_generator.num_rows = n
            self.scene.terrain.terrain_generator.num_cols = n
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.max_init_terrain_level = None

        # disable randomization for play

        self.observations.policy.enable_corruption = False

        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
