# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

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

        # set robot asset and scanner prim path
        # 设置场景中使用的机器人资源以及高度扫描器（height_scanner）所绑定的 prim 路径
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
        # 禁用推机器人事件以避免在该配置下引入随机扰动（此处选择关闭）
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        # 调整为更适合小机器人基座质量扰动范围（负表示减少质量，正表示增加）
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        # 指定只影响基座（base）
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        # 外力事件也只影响基座
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # 重置关节时使用固定位置比例（缩放为 1.0 表示使用默认初始化范围）
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
        # 重置基座位姿与速度：位姿在指定范围内随机化，速度设为 0（基座不带初始速度）
        self.events.base_com = None
        # 禁用基座质心（COM）随机化事件，以保持基座动力学更稳定

        # rewards
        # 奖励项的调整（针对小机器人与训练目标进行微调）
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        # 将 feet_air_time 的传感器 body_names 匹配改为小写或其他命名习惯的脚部标识（正则）
        self.rewards.feet_air_time.weight = 0.01
        # 缩小脚部空中时间奖励权重，避免过度鼓励大幅抬脚（对小机器人更谨慎）
        self.rewards.undesired_contacts = None
        # 移除“不期望接触”惩罚（例如 thigh 接触），在此配置中不需要或会误触发
        self.rewards.dof_torques_l2.weight = -0.0002
        # 调整关节力矩惩罚权重（能耗项）
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        # 增加线速度跟踪奖励权重以强化跟踪任务
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        # 调整角速度跟踪奖励权重
        self.rewards.dof_acc_l2.weight = -2.5e-7
        # 关节加速度惩罚权重（平滑动作）

        # terminations
        # 终止条件配置调整
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        # 将基座接触检测限制为 prim 名称中包含 base 的刚体，确保基座接触触发终止逻辑按预期工作


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
