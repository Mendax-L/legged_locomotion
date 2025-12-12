# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
# 预定义配置
# (保留英文注释) 以上导入预定义的 rough 地形配置以供场景使用。

##
# Scene definition
##


@configclass
class LeggedLocomotionCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""
    # Configuration for the terrain scene with a legged robot.
    # 带有步态机器人的地形场景配置。

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # ground terrain
    # 地面地形配置（使用 TerrainImporterCfg，生成器类型，纹理与物理属性已配置）

    # robots
    robot: ArticulationCfg = MISSING
    # robots
    # 机器人（必填配置，占位为 MISSING）

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # sensors
    # 传感器配置：高度扫描器（ray caster），在机器人 base 上方，网格模式，分辨率与尺寸已设置

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # contact_forces
    # 接触力传感器配置，记录历史、跟踪空中时间

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    # lights
    # 天空光源配置（半球光），强度与贴图路径已配置


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    # Command specifications for the MDP.
    # MDP 的命令规范配置。

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.5, 5.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )
    # base_velocity
    # 基础速度命令配置，包含线速度与角速度范围以及重采样时间等参数


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Action specifications for the MDP.
    # MDP 的动作规范配置。

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)}
    )
    # joint_pos
    # 关节位置动作配置：对所有关节应用，缩放因子为 0.5，使用默认偏移


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    # Observation specifications for the MDP.
    # MDP 的观测规范配置。


    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # Observations for policy group.
        # 策略组的观测项配置（各项顺序保留）。

        # # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        # # base_lin_vel
        # # 基础线速度观测项，添加均匀噪声扰动
        # 现实中不观测线速度，噪声很大

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # base_ang_vel
        # 基础角速度观测项，添加均匀噪声扰动

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # projected_gravity
        # 投影重力观测项，带噪声

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # velocity_commands
        # 速度命令观测项（来自命令管理器）

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # joint_pos
        # 相对关节位置观测项，带小幅噪声

        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # joint_vel
        # 相对关节速度观测项，带较大噪声范围

        actions = ObsTerm(func=mdp.last_action)
        # actions
        # 上一步动作观测项

        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )
        # 高度扫描观测项（RayCaster），绑定到 height_scanner 传感器，含噪声与裁剪范围

        def __post_init__(self):
            self.enable_corruption = True
            # 拼接的影响可做测试
            self.concatenate_terms = True
        # __post_init__
        # 后处理：启用观测损坏（corruption）并将各项拼接为单一观测向量
        
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group (can have more privileged information)."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


    # observation groups
    # 观测组：包含 policy 组


@configclass
class EventCfg:
    """Configuration for events."""
    # Configuration for events.
    # 事件（例如启动、重置、周期性）的配置集合。

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # physics_material
    # 启动时随机化刚体材料属性（摩擦、回复等）

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    # add_base_mass
    # 启动时为基座添加质量扰动

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )
    # base_com
    # 启动时随机化质心位置

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    # base_external_force_torque
    # 重置时应用外力/力矩（此处范围设为 0）

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    # reset_base
    # 重置根部位姿与速度，范围已配置

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    # reset_robot_joints
    # 重置关节位置与速度的比例范围

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    # push_robot
    # 周期性事件：给予机器人推力模拟干扰，间隔范围已设


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # Reward terms for the MDP.
    # 奖励项配置（任务奖励与惩罚项）。

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # track_lin_vel_xy_exp
    # 任务：跟踪线速度（xy），指数核，作为主要奖励

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # track_ang_vel_z_exp
    # 任务：跟踪角速度（偏航），指数核，作为次要奖励

    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    # lin_vel_z_l2
    # 惩罚：竖直方向线速度 L2（避免跳跃）

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1.0)
    # ang_vel_xy_l2
    # 惩罚：平面内滚摆角速度 L2（保持稳定）

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0)
    # dof_torques_l2
    # 惩罚：关节扭矩 L2（鼓励省能量）

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0)
    # dof_acc_l2
    # 惩罚：关节加速度 L2（平滑动作）

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0)
    # action_rate_l2
    # 惩罚：动作变化率 L2（减少频繁动作变化）

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.25,
        },
    )
    # feet_air_time
    # 奖励：脚的空中时间（鼓励迈步），使用接触力传感器作为输入

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh"), "threshold": 1.0},
    )
    # undesired_contacts
    # 惩罚：不希望的接触（如大腿与地面接触）

    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    base_height_l2 = RewTerm(
        func=mdp.safe_base_height_l2,
        weight=-1.0,
        params={"target_height": 0.32, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    body_lin_acc_l2 = RewTerm(func=mdp.body_lin_acc_l2, weight=-1.0)

    # style
    encourage_forward = RewTerm(
        func=mdp.encourage_forward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


    cheetah = RewTerm(
        func=mdp.encourage_default_pose,
        weight=1.0,
        params={"hip_weight": 1.0, "thigh_weight": 0, "calf_weight": 0, "asset_cfg": SceneEntityCfg("robot")}
    )

    velocity_driven_gait = RewTerm(
        func=mdp.velocity_driven_gait,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # Termination terms for the MDP.
    # 终止条件配置（例如超时或非法接触）。

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # time_out
    # 超时终止项

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    # base_contact
    # 基座非法接触终止（例如基座碰撞地面）


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # Curriculum terms for the MDP.
    # 课程（难度进度）相关配置项。

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # terrain_levels
    # 地形难度随训练进度动态调整的课程项


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Configuration for the locomotion velocity-tracking environment.
    # 行走速度跟踪环境的总体配置。

    # Scene settings
    scene: LeggedLocomotionCfg = LeggedLocomotionCfg(num_envs=4096, env_spacing=2.5)
    # Scene settings
    # 场景设置：环境数量与间距

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # Basic settings
    # 基本设置：观测、动作与命令配置

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # MDP settings
    # MDP 相关设置：奖励、终止、事件、课程

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # general settings
        # 通用设置：降采样因子与每集时长

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_patch_count = 2 ** 26
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 18 #重要！！！
        # simulation settings
        # 仿真设置：物理步长、渲染间隔、物理材质、PhysX 参数

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # update sensor update periods
        # 更新传感器的更新周期：基于最小更新周期（物理步长）统一设置

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
        # check curriculum flag
        # 检查是否启用地形等级课程并相应设置 terrain_generator 的 curriculum 字段
