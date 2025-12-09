import time

import numpy as np
import torch
from pathlib import Path

from deployment.utils.gamepad import Gamepad
from deployment.interface import Go2_mujoco, Go2_real
from deployment.config import Config


class BaseController:
    def __init__(self, cfg: Config) -> None:

        self.cfg = cfg
        self._robot = Go2_real(self.cfg) if self.cfg.is_real else Go2_mujoco(self.cfg)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        if self.cfg.use_gamepad:
            self.gp = Gamepad()
            self.gp.start()

        if not self.gp.connected:
            self.cfg.use_gamepad = False

        self.obs = np.zeros(self.cfg.num_obs)
        self.action = np.zeros(self.cfg.num_actions)
        self.raw_action = np.zeros(self.cfg.num_actions)
        self.dt = self.cfg.control_dt
        self.init_pos = self.cfg.init_pos[self.cfg.sim2real]
        self.default_pos = self.cfg.default_pos[self.cfg.sim2real]

        self.policy = torch.jit.load(self.cfg.model_path, map_location=self.device)
        self.policy.eval()

    def load_policy(self, obs):
        obs_dict = {
            "obs": torch.as_tensor(obs["obs"], dtype=torch.float).unsqueeze(0).to(self.device),
        }
        action = self.policy(obs_dict["obs"])[0]
        return action

    def processing_action(self, raw_action):
        raw_action_clipped = np.clip(raw_action, -self.cfg.action_clip, self.cfg.action_clip).copy()  # clip
        # isaac lab [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]
        # mujoco [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
        action_mujoco = raw_action_clipped[self.cfg.sim2real]  # change index
        action_scaled = action_mujoco * self.cfg.action_scale  # scaled
        tar_dof_pos = action_scaled + self.default_pos  # relative to absolute

        return tar_dof_pos

    def processing_observation(
            self,
            base_ang_vel,
            projected_gravity,
            command,
            dof_pos,
            dof_vel,
            last_action,
    ):
        # mujoco [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
        # isaac lab [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]
        dof_pos = dof_pos - self.default_pos  # relative position
        dof_pos_isaac = dof_pos[self.cfg.real2sim]
        dof_vel_isaac = dof_vel[self.cfg.real2sim]

        obs = np.hstack(
            (
                base_ang_vel,  # 3
                projected_gravity,  # 3
                command,  # 3
                dof_pos_isaac,  # 12
                dof_vel_isaac,  # 12
                last_action,  # 12
            )
        )

        return {'obs': obs}

    def run(self):
        """from init to standing to running to stopping"""
        if self.cfg.use_gamepad:
            print("waiting LB + A ...")
            while True:
                if self.gp.a_pressed and self.gp.lb_pressed:
                    break
        self.stand_up()

        while True:
            self.step(self.cfg.kp, self.cfg.kd, self.action)
            # np.savez(f"./data/temp_log_gamepad.npz", **self._robot.logger)

            if self.cfg.use_gamepad:
                if self.gp.lb_pressed and self.gp.a_pressed:
                    self.stand_down()
                    break
                if self.gp.b_pressed:
                    break

        self.stop()
        self.gp.stop()

        print("saving data...")
        time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if not Path(f'./data/').exists():
            Path('./data/').mkdir()
        np.savez(f"./data/{time_now}.npz", **self._robot.logger)

        print("logs saving over!")

    def step(self, kp, kd, motor_cmd):
        self._robot.set_motor_command(kp, kd, motor_cmd)
        self._robot.step()

        command = self.get_gamepad_command()
        self.obs = self.processing_observation(
            base_ang_vel=np.array(self._robot.base_ang_vel),
            projected_gravity=np.array(self._robot.projected_gravity),
            command=np.array(command),
            dof_pos=np.array(self._robot.dof_pos),
            dof_vel=np.array(self._robot.dof_vel),
            last_action=np.array(self.raw_action),
        )

        self.raw_action = self.load_policy(self.obs).reshape(-1).cpu().detach().numpy()
        self.action = self.processing_action(self.raw_action)

    def stand_up(self):
        stand_time = 5
        wait_time = 5

        self._robot.set_motor_command(0.0, 0.0, self.init_pos)
        self._robot.step()
        current_joint_angle = np.array(self._robot.dof_pos)

        for t in np.arange(0, stand_time, self.dt):
            blend_ratio = min(t / stand_time, 1)
            action = blend_ratio * self.default_pos + (1 - blend_ratio) * current_joint_angle
            self.step(self.cfg.stand_kp, self.cfg.stand_kd, action)
            self.raw_action = np.zeros(self.cfg.num_actions)

            if self.cfg.use_gamepad and self.gp.b_pressed:
                self.stop()
                self.gp.stop()
                print("Emergency Stop!")
                exit(0)

        for _ in np.arange(0, wait_time, self.dt):
            self.step(self.cfg.stand_kp, self.cfg.stand_kd, self.default_pos)  # 稳定在default position
            self.raw_action = np.zeros(self.cfg.num_actions)

            if self.cfg.use_gamepad and self.gp.b_pressed:
                self.stop()
                self.gp.stop()
                print("Emergency Stop!")
                exit(0)

    def stand_down(self):
        stand_time = 3
        self._robot.set_motor_command(0.0, 0.0, np.zeros(self.cfg.num_actions))
        self._robot.step()
        current_joint_angle = np.array(self._robot.dof_pos)

        for t in np.arange(0, stand_time, self.dt):
            blend_ratio = min(t / stand_time, 1)
            action = blend_ratio * self.init_pos + (1 - blend_ratio) * current_joint_angle
            self.step(self.cfg.stand_kp, self.cfg.stand_kd, action)
            self.raw_action = np.zeros(self.cfg.num_actions)

            if self.cfg.use_gamepad and self.gp.b_pressed:
                self.stop()
                self.gp.stop()
                print("Emergency Stop!")
                exit(0)

    def stop(self):
        stop_time = 10
        for _ in np.arange(0, stop_time, self.dt):
            self._robot.set_motor_command(0.0, 5.0, np.zeros(self.cfg.num_actions))
            self._robot.step()

    def get_gamepad_command(self):
        if not self.cfg.fixed_command and self.gp.connected:
            return np.array([self.gp.vel_x, self.gp.vel_y, 0, self.gp.vel_yaw])
        else:
            return self.cfg.command
