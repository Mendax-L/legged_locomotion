# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from legged_locomotion.tasks.manager_based.legged_locomotion.leggedlocomotion_velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
# Pre-defined configs
# 预定义配置
# 导入单个已定义好的机器人配置（UNITREE_GO2_CFG），用于在下面的环境配置中设置具体机器人模型。
# isort: skip 保持此导入位置以避免自动排序工具改变导入顺序（可能因依赖关系要求）。

@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        # 调用父类的后初始化以继承并设置基础配置
        super().__post_init__()
        import inspect, os

        # set robot asset and scanner prim path
        # 设置场景中使用的机器人资源以及高度扫描器（height_scanner）所绑定的 prim 路径
        # self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



        # self.scene.robot = UNITREE_GO2_CFG.replace(
        #     prim_path="{ENV_REGEX_NS}/Robot",
        #     spawn=UNITREE_GO2_CFG.spawn.replace(
        #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #             enabled_self_collisions=False
        #         )
        #     ),
        # )

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



        # 将导入的 UNITREE_GO2_CFG 的 prim_path 替换为环境命名空间下的 Robot 路径（每个 env 有自己的命名空间）
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # 将高度扫描器绑定到机器人 base 上，确保扫描器在每个环境中正确定位

        # scale down the terrains because the robot is small
        # 缩小地形参数以适配体型更小的机器人（Go2），避免地形过大导致不合理的交互尺度
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # boxes 子地形高度范围调整为较小值
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        # random_rough 噪声幅度调小
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        # 噪声步长（细化地形细节的粒度）

        # reduce action scale
        # 缩小动作尺度（关节位置命令的缩放），适配 Go2 的舵机/控制范围
        self.actions.joint_pos.scale = 0.25

        # event
        # 事件（Event）相关调整
        self.events.push_robot = None
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

        # penalties
        self.rewards.lin_vel_z_l2.weight = -5.0
        self.rewards.ang_vel_xy_l2.weight = -1e-1
        self.rewards.dof_torques_l2.weight = -5e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -1.0
        self.rewards.undesired_contacts.weight = -1e3
        self.rewards.flat_orientation_l2.weight = -50.0
        self.rewards.dof_pos_limits.weight = -5e2
        self.rewards.base_height_l2.weight = -1e2
        self.rewards.body_lin_acc_l2.weight = -5e-4

        # style
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.encourage_forward.weight = 2.0
        self.rewards.speed_limit.weight = 1.0
        self.rewards.cheetah.weight = 3.0
        self.rewards.velocity_driven_gait.weight = 5.0

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # curriculums
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None

@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        # 调用父类后初始化以继承上面的所有设置
        super().__post_init__()

        # make a smaller scene for play
        # 为交互式“play”模式创建更小规模的场景，方便调试与可视化
        self.scene.num_envs = 50
        # 环境数量减少，便于交互式观察训练效果
        self.scene.env_spacing = 2.5
        # 环境间距保持一致

        # spawn the robot randomly in the grid (instead of their terrain levels)
        # 在网格内随机生成机器人，而不是依据 terrain_levels 来放置（play 模式更随机）
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        # 减少生成的子地形数量以节省内存与加速渲染（play 模式）
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
            # 将地形网格缩小为 5x5，并禁用地形难度课程（curriculum），以固定地形集合便于调试

        # disable randomization for play
        # 在 play 模式关闭观测扰动与随机化，便于一致性测试与调试
        self.observations.policy.enable_corruption = False

        # remove random pushing event
        # 在 play 模式禁用外力与推力事件，避免随机干扰影响可视化调试
        self.events.base_external_force_torque = None
        self.events.push_robot = None
