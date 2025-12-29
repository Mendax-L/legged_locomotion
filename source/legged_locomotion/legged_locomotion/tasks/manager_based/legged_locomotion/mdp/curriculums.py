# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""
# 可用于为学习环境创建课程（难度进度）的通用函数。
#
# 这些函数可以传递给 isaaclab.managers.CurriculumTermCfg 来启用由函数引入的课程规则。

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # 基于机器人在期望速度命令下行走的距离来调整课程（地形难度）。
    #
    # 当机器人行走足够远时增加地形难度；当行走距离小于命令速度要求的一半时降低难度。
    #
    # 注意：该项仅适用于地形类型为 "generator" 的情况。有关地形类型的更多信息，请参见 isaaclab.terrains.TerrainImporter。

    # extract the used quantities (to enable type-hinting)
    # 提取所需量（以启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    # 计算机器人行走的距离
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    # 行走足够远的机器人进入更难的地形
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    # 行走距离小于所需距离一半的机器人降级到更简单的地形
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    # 更新地形难度等级
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    # 返回平均地形等级
    return torch.mean(terrain.terrain_levels.float())
