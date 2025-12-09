# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for ANYmal.
用于为 ANYmal 指定观测与动作空间的对称性变换的函数。
"""
s
from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
# 指定可被导入的函数
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.

    对观测和动作应用对称变换以生成增强数据。

    该函数对输入的观测与动作分别生成四种变换：原始、左右对称、前后对称、以及对角（左右+前后）。
    这些对称增强可在强化学习中增加数据多样性而无需额外采集数据。

    参数:
        env: 环境实例。
        obs: 原始观测的 TensorDict，可选。
        actions: 原始动作的张量，可选。

    返回:
        增强后的观测和动作（如果输入为 None，则对应返回 None）。
    """

    # observations
    # 观测处理
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        # 因为有 4 种对称性，要将批次扩增为 4 倍
        obs_aug = obs.repeat(4)

        # policy observation group
        # -- original
        # 策略组观测
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])
        # -- front-back
        obs_aug["policy"][2 * batch_size : 3 * batch_size] = _transform_policy_obs_front_back(
            env.unwrapped, obs["policy"]
        )
        # -- diagonal
        obs_aug["policy"][3 * batch_size :] = _transform_policy_obs_front_back(
            env.unwrapped, obs_aug["policy"][batch_size : 2 * batch_size]
        )
    else:
        obs_aug = None

    # actions
    # 动作处理
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        # 同样将动作批次扩增为 4 倍
        actions_aug = torch.zeros(batch_size * 4, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
        # -- front-back
        actions_aug[2 * batch_size : 3 * batch_size] = _transform_actions_front_back(actions)
        # -- diagonal
        actions_aug[3 * batch_size :] = _transform_actions_front_back(actions_aug[batch_size : 2 * batch_size])
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
对观测的对称变换函数。
"""


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    negating certain components of the linear and angular velocities, projected gravity,
    velocity commands, and flipping the joint positions, joint velocities, and last actions
    for the ANYmal robot. Additionally, if height-scan data is present, it is flipped
    along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.

    对观测张量应用左右对称变换。

    该函数对线速度、角速度、投影重力、速度指令的某些分量取反，并对关节位置、关节速度、上一步动作进行左右交换。
    若存在 height_scan（高度扫描）数据，也会沿相应维度翻转。

    参数:
        env: 获取观测的环境实例。
        obs: 需要变换的观测张量。

    返回:
        应用了左右对称的观测张量。
    """
    # copy observation tensor
    # 复制观测张量（避免原地修改）
    obs = obs.clone()
    device = obs.device
    # lin vel
    # 线速度
    obs[:, :3] = obs[:, :3] * torch.tensor([1, -1, 1], device=device)
    # ang vel
    # 角速度
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    # 投影重力
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)
    # velocity command
    # 速度指令
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    # 关节位置
    obs[:, 12:24] = _switch_anymal_joints_left_right(obs[:, 12:24])
    # joint vel
    # 关节速度
    obs[:, 24:36] = _switch_anymal_joints_left_right(obs[:, 24:36])
    # last actions
    # 上一步动作
    obs[:, 36:48] = _switch_anymal_joints_left_right(obs[:, 36:48])

    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    # 注意：以下对 height_scan 的处理基于硬编码，假设网格按 "xy" 排序且尺寸为 (1.6, 1.0)
    if "height_scan" in env.observation_manager.active_terms["policy"]:
        obs[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[1]).view(-1, 11 * 17)

    return obs


def _transform_policy_obs_front_back(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the front-back axis. This includes negating
    certain components of the linear and angular velocities, projected gravity, velocity commands,
    and flipping the joint positions, joint velocities, and last actions for the ANYmal robot.
    Additionally, if height-scan data is present, it is flipped along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with front-back symmetry applied.

    对观测张量应用前后对称变换。

    该函数对线速度、角速度、投影重力、速度指令的某些分量取反，并对关节位置、关节速度、上一步动作进行前后交换。
    若存在 height_scan（高度扫描）数据，也会沿相应维度翻转。

    参数:
        env: 获取观测的环境实例。
        obs: 需要变换的观测张量。

    返回:
        应用了前后对称的观测张量。
    """
    # copy observation tensor
    # 复制观测张量（避免原地修改）
    obs = obs.clone()
    device = obs.device
    # lin vel
    # 线速度
    obs[:, :3] = obs[:, :3] * torch.tensor([-1, 1, 1], device=device)
    # ang vel
    # 角速度
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, -1], device=device)
    # projected gravity
    # 投影重力
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([-1, 1, 1], device=device)
    # velocity command
    # 速度指令
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([-1, 1, -1], device=device)
    # joint pos
    # 关节位置
    obs[:, 12:24] = _switch_anymal_joints_front_back(obs[:, 12:24])
    # joint vel
    # 关节速度
    obs[:, 24:36] = _switch_anymal_joints_front_back(obs[:, 24:36])
    # last actions
    # 上一步动作
    obs[:, 36:48] = _switch_anymal_joints_front_back(obs[:, 36:48])

    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    # 注意：以下对 height_scan 的处理基于硬编码，假设网格按 "xy" 排序且尺寸为 (1.6, 1.0)
    if "height_scan" in env.observation_manager.active_terms["policy"]:
        obs[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[2]).view(-1, 11 * 17)

    return obs


"""
Symmetry functions for actions.
对动作的对称变换函数。
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.

    对动作张量应用左右对称变换。

    该函数对动作中的关节分量进行左右交换，并对需要取反的关节分量取反（例如 HAA）。
    """
    actions = actions.clone()
    actions[:] = _switch_anymal_joints_left_right(actions[:])
    return actions


def _transform_actions_front_back(actions: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the front-back axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with front-back symmetry applied.

    对动作张量应用前后对称变换。

    该函数对动作中的关节分量进行前后交换，并对需要取反的关节分量取反（例如 HFE、KFE）。
    """
    actions = actions.clone()
    actions[:] = _switch_anymal_joints_front_back(actions[:])
    return actions


"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[
    'LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA',
    'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE',
    'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE'
]

Correspondingly, the joint ordering for the ANYmal robot is:

* LF = left front --> [0, 4, 8]
* LH = left hind --> [1, 5, 9]
* RF = right front --> [2, 6, 10]
* RH = right hind --> [3, 7, 11]

辅助对称函数说明。

在 Isaac Sim 中关节顺序如下（英文顺序保留）：
[
    'LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA',
    'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE',
    'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE'
]

对应的 ANYmal 关节索引映射说明（中文）：
* LF = 左前  --> [0, 4, 8]
* LH = 左后  --> [1, 5, 9]
* RF = 右前  --> [2, 6, 10]
* RH = 右后  --> [3, 7, 11]
"""


def _switch_anymal_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    # 对关节数据应用左右对称变换
    joint_data_switched = torch.zeros_like(joint_data)
    # left <-- right
    # 左侧接收右侧的数据
    joint_data_switched[..., [0, 4, 8, 1, 5, 9]] = joint_data[..., [2, 6, 10, 3, 7, 11]]
    # right <-- left
    # 右侧接收左侧的数据
    joint_data_switched[..., [2, 6, 10, 3, 7, 11]] = joint_data[..., [0, 4, 8, 1, 5, 9]]

    # Flip the sign of the HAA joints
    # 对 HAA 关节（髋关节横向）取反符号
    joint_data_switched[..., [0, 1, 2, 3]] *= -1.0

    return joint_data_switched


def _switch_anymal_joints_front_back(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the joint data tensor."""
    # 对关节数据应用前后对称变换
    joint_data_switched = torch.zeros_like(joint_data)
    # front <-- hind
    # 前侧接收后侧的数据
    joint_data_switched[..., [0, 4, 8, 2, 6, 10]] = joint_data[..., [1, 5, 9, 3, 7, 11]]
    # hind <-- front
    # 后侧接收前侧的数据
    joint_data_switched[..., [1, 5, 9, 3, 7, 11]] = joint_data[..., [0, 4, 8, 2, 6, 10]]

    # Flip the sign of the HFE and KFE joints
    # 对 HFE 与 KFE 关节（膝/腿屈伸类）取反符号
    joint_data_switched[..., 4:] *= -1

    return joint_data_switched
