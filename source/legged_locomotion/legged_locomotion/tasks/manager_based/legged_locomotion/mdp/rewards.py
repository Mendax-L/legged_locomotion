# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, quat_apply, quat_conjugate, normalize

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
        env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward



def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def velocity_tracking_xy(
        env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std ** 2)

def velocity_tracking_yaw(
        env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std ** 2)








def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()


def heading_command_error_abs_exp(
        env: ManagerBasedRLEnv,
        command_name: str,
        scale: float = 0.25
) -> torch.Tensor:

    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]  # heading error in body frame

    reward = torch.exp(-heading_b.abs() / scale)

    return reward


def speed_limit(
        env,
        threshold: float = 1.0,
        scale: float = 0.25,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_speed_xy = torch.norm(vel_yaw[:, :2], dim=1)  # speed in yaw-aligned frame
    speed_excess = torch.clamp(lin_speed_xy - threshold, min=0.0)
    return torch.exp(-speed_excess / scale) - 1


def encourage_forward(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_w = asset.data.root_lin_vel_w[:, :2]  # (B, 2)
    quat = asset.data.root_quat_w  # (B, 4)

    batch_size = quat.shape[0]
    local_forward = torch.tensor([1, 0, 0],
                                 dtype=torch.float32,
                                 device=quat.device).reshape(1, 3).expand(batch_size, -1)

    fwd_dir_world = quat_apply(quat, local_forward)[:, :2]
    fwd_dir_world = fwd_dir_world / (torch.norm(fwd_dir_world, dim=1, keepdim=True) + 1e-8)

    forward_speed = torch.sum(vel_w * fwd_dir_world, dim=1)
    speed_norm = torch.norm(vel_w, dim=1)
    alignment = forward_speed / (speed_norm + 1e-8)
    
    if not torch.isfinite(alignment).all():         # 新增
        print("encourage_forward 出现 inf!", alignment)
    return alignment


def encourage_forward_linear_only(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """
    奖励机器人在世界系中 x 方向速度为正，y 方向速度尽量为 0 的行为。
    用于导航任务，鼓励直线前行，抑制侧向滑动。
    """
    asset = env.scene[asset_cfg.name]
    vel_w = asset.data.root_lin_vel_w[:, :2]  # [B, 2], x-y 平面速度

    vx = vel_w[:, 0]  # x 方向速度
    vy = vel_w[:, 1]  # y 方向速度

    # 奖励正向速度，惩罚侧向偏移
    reward = vx - 0.5 * torch.abs(vy)  # 惩罚系数 0.5 可调
    return reward


def encourage_default_pose(
        env: ManagerBasedRLEnv,
        hip_weight: float = 1.0,
        thigh_weight: float = 0.0,
        calf_weight: float = 0.0,
        speed_threshold: float = 0.1,  # 速度阈值 (m/s)
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Encourage the robot to stay close to its default joint configuration.
    - When the robot speed < threshold: encourage *all* joints to stay near default.
    - When the robot speed >= threshold: use standard weighted reward.
    """
    asset = env.scene[asset_cfg.name]
    dof_pos = asset.data.joint_pos  # (num_envs, 12)
    default_dof_pos = asset.data.default_joint_pos  # (num_envs, 12)
    body_lin_vel = asset.data.root_lin_vel_w[:, :2]  # (num_envs, 2)

    # Step 1: Compute body planar speed
    speed = torch.norm(body_lin_vel, dim=1)  # (num_envs,)

    # Step 2: 构建原始加权reward（运动时使用）
    weights_active = torch.tensor(
        [hip_weight] * 4 + [thigh_weight] * 4 + [calf_weight] * 4,
        device=dof_pos.device
    ).unsqueeze(0)  # (1, 12)

    error_active = weights_active * (dof_pos - default_dof_pos) ** 2
    reward_active = torch.exp(-torch.sum(error_active, dim=1))  # (num_envs,)

    # Step 3: 构建静止reward（所有关节同等参与）
    weights_static = torch.ones_like(dof_pos, device=dof_pos.device)
    error_static = weights_static * (dof_pos - default_dof_pos) ** 2
    reward_static = torch.exp(-torch.sum(error_static, dim=1))  # (num_envs,)

    # Step 4: 速度判断，分段奖励
    is_static = speed < speed_threshold
    reward = torch.where(is_static, reward_static, reward_active)

    if not torch.isfinite(reward).all():         # 新增
        print("encourage_default_pose 出现 inf!", reward)
    return reward


def raibert_heuristic(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Raibert heuristic-based reward encouraging foot placement alignment with current velocity and yaw rate.

    This variant estimates the desired footstep location using the Raibert heuristic, based on the robot's
    current linear and angular velocity. It assumes a fixed gait frequency and does not require command inputs.

    Returns:
        A positive reward (the smaller the error, the higher the reward).
    """
    asset = env.scene[asset_cfg.name]

    # -- base state
    base_pos = asset.data.root_pos_w  # [N, 3]
    base_quat = asset.data.root_quat_w  # [N, 4]
    lin_vel = asset.data.root_lin_vel_w  # [N, 3]
    ang_vel = asset.data.root_ang_vel_w  # [N, 3]

    # -- foot positions in world, convert to body frame
    foot_ids = [i for i, name in enumerate(asset.body_names) if "foot" in name.lower()]
    foot_world = asset.data.body_link_pos_w[:, foot_ids, :]
    foot_rel = foot_world - base_pos.unsqueeze(1)  # [N, 4, 3]
    foot_body = quat_apply_yaw_broadcast(base_quat, foot_rel)

    # -- nominal stance (assumed fixed geometry)
    stance_length = 0.40
    stance_width = 0.30
    xs_nom = torch.tensor([+stance_length / 2, +stance_length / 2, -stance_length / 2, -stance_length / 2],
                          device=env.device).unsqueeze(0)  # [1, 4]
    ys_nom = torch.tensor([+stance_width / 2, -stance_width / 2, +stance_width / 2, -stance_width / 2],
                          device=env.device).unsqueeze(0)  # [1, 4]

    # -- gait parameters (assume fixed 2.0Hz frequency)
    frequency = 2.0
    freq = torch.full((env.num_envs, 1), frequency, device=env.device)  # [N, 1]
    phase = torch.tensor([[+0.5], [-0.5], [+0.5], [-0.5]], device=env.device).T  # [1, 4]

    # -- velocity estimates
    x_vel_des = lin_vel[:, 0:1]  # [N, 1]
    yaw_vel_des = ang_vel[:, 2:3]  # [N, 1]
    y_vel_des = yaw_vel_des * stance_length / 2  # [N, 1]

    # -- Raibert offset
    xs_offset = phase * x_vel_des * (0.5 / freq)  # [N, 4]
    ys_offset = phase * y_vel_des * (0.5 / freq)  # [N, 4]
    ys_offset[:, 2:4] *= -1  # reverse rear legs offset

    # -- desired position in body frame
    xs_target = xs_nom + xs_offset
    ys_target = ys_nom + ys_offset
    foot_target_body = torch.stack([xs_target, ys_target], dim=2)  # [N, 4, 2]

    # -- actual position in body frame
    foot_actual_body = foot_body[:, :, :2]  # [N, 4, 2]

    # -- squared error
    error = torch.square(foot_target_body - foot_actual_body)  # [N, 4, 2]
    reward = -torch.sum(error, dim=(1, 2))  # [N], negative of error as reward

    return reward


# utils
def quat_apply_yaw_broadcast(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    N, M, _ = vec.shape
    quat_rep = quat_conjugate(quat).unsqueeze(1).expand(-1, M, -1).reshape(-1, 4)
    quat_rep[:, :2] = 0.0
    quat_yaw = normalize(quat_rep)
    vec_flat = vec.reshape(-1, 3)
    rotated_flat = quat_apply(quat_yaw, vec_flat)
    rotated = rotated_flat.view(N, M, 3)

    return rotated


def safe_base_height_l2(
        env: ManagerBasedRLEnv,
        target_height: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height

    # Compute the L2 squared penalty
    reward =  torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height).clamp(-100.0, 100.0)
    if not torch.isfinite(reward).all():         # 新增
        print("safe_base_height_l2 出现 inf!", reward)
    return reward
# mdp/rewards.py  末尾追加

def velocity_driven_gait(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        gait_sharpness: float = 0.25,
        speed_upper_bound: float = 1.5,
) -> torch.Tensor:
    """
    根据实际身体速度决定 gait reward 插值权重（越快越偏向bound ，越慢越偏向trot）。

    参数:
    - gait_sharpness: float, 控制 reward 对误差的敏感程度，值越大，reward 趋近于 0/1 阶跃型。
    - speed_upper_bound: float, 控制 gait 插值的速度最大值（单位 m/s），用于归一化速度。
    """
    actions = env.action_manager.action  # [num_envs, 12]

    body_lin_vel = env.scene[asset_cfg.name].data.root_lin_vel_w[:, :3]  # [num_envs, 3] in world frame

    # Step 1: Normalize body speed
    speed = torch.norm(body_lin_vel[:, :2], dim=1)  # [num_envs]
    speed_normalized = torch.clamp(speed / speed_upper_bound, 0.0, 1.0)
    lambda_bound = speed_normalized
    lambda_trot = 1.0 - lambda_bound

    # Step 2: Gait-specific reward terms
    # --- (a) Trot reward: diagonal leg symmetry ---
    # FL vs RR: hip, thigh, calf => indices [0,4,8] vs [3,7,11]
    # FR vs RL: hip, thigh, calf => indices [1,5,9] vs [2,6,10]
    trot_error = torch.square(actions[:, [0, 4, 8]] - actions[:, [3, 7, 11]]).mean(dim=1) + \
                 torch.square(actions[:, [1, 5, 9]] - actions[:, [2, 6, 10]]).mean(dim=1)
    reward_trot = torch.exp(-gait_sharpness * trot_error)

    # --- (b) Bound reward: front vs. back leg symmetry ---
    # FL vs RR: hip, thigh, calf => indices [0,4,8] vs [1,5,9]
    # FR vs RL: hip, thigh, calf => indices [2,6,10] vs [3,7,11]
    # Thigh + calf (FL vs FR, RL vs RR) → 差为 0
    leg_diff = (
            torch.square(actions[:, [4, 8]] - actions[:, [5, 9]]).mean(dim=1) +  # front legs (thigh+calf)
            torch.square(actions[:, [6, 10]] - actions[:, [7, 11]]).mean(dim=1)  # rear legs (thigh+calf)
    )
    # Hip Opposite
    hip_opposite = (
            torch.square(actions[:, 0] + actions[:, 1]) +  # front hip
            torch.square(actions[:, 2] + actions[:, 3])  # rear hip
    )
    # Final bound error
    bound_error = leg_diff + hip_opposite
    reward_bound = torch.exp(-gait_sharpness * bound_error)

    # --- (c) Consistent reward -> front vs. back leg consistent
    # front legs: FL+FR = [0,1,4,5,8,9], back legs RL+RR = [2,3,6,7,10,11]
    front_leg_avg = actions[:, [0, 1, 4, 5, 8, 9]].mean(dim=1)
    back_leg_avg = actions[:, [2, 3, 6, 7, 10, 11]].mean(dim=1)
    consistent_error = torch.square(front_leg_avg - back_leg_avg)
    reward_consistent = torch.exp(-gait_sharpness * consistent_error)

    # Step 3: Interpolated reward
    # reward = lambda_trot * reward_trot + lambda_bound * reward_bound + reward_consistent
    reward = 1 * reward_trot + 0 * reward_bound + reward_consistent
    if not torch.isfinite(reward).all():         # 新增
        print("velocity_driven_gait 出现 inf!", reward)
    return reward  # shape: [num_envs]




# def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Reward long steps taken by the feet for bipeds.
#     This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
#     a time in the air.
#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     """
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
#     contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
#     in_contact = contact_time > 0.0
#     in_mode_time = torch.where(in_contact, contact_time, air_time)
#     single_stance = torch.sum(in_contact.int(), dim=1) == 1
#     reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
#     reward = torch.clamp(reward, max=threshold)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     return reward

# def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
#     """Reward position tracking with tanh kernel."""
#     command = env.command_manager.get_command(command_name)
#     des_pos_b = command[:, :3]
#     distance = torch.norm(des_pos_b, dim=1)
#     return 1 - torch.tanh(distance / std)


# def position_command_error_exp(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
#     """Reward position tracking with tanh kernel."""
#     command = env.command_manager.get_command(command_name)
#     des_pos_b = command[:, :3]
#     distance = torch.norm(des_pos_b, dim=1)
#     return torch.exp(-distance / std)

# def delta_position_command_exp(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
#     command = env.command_manager.get_command(command_name)
#     now_cmd = command[:, :3]
#     last_cmd = env.command_manager._terms["pose_command"].last_pos_command_b
#     # distance = torch.clamp(torch.norm(last_cmd, dim=1) - torch.norm(now_cmd, dim=1), max=0.2)
#     distance = torch.norm(last_cmd, dim=1) - torch.norm(now_cmd, dim=1)
#     return distance