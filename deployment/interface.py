"""
@Author: Yang Xuekang
@E-mail: yangxuekang@sjtu.edu.cn
@Date  : 2025/11/3
"""

import time
import warnings

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from deployment.utils.logger import SimpleLogger
import mujoco.viewer
from deployment.config import Config


class Go2_mujoco:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.init_pos = self.cfg.init_pos[self.cfg.sim2real]
        self.default_pos = self.cfg.default_pos[self.cfg.sim2real]
        self.kp = self.cfg.stand_kp
        self.kd = self.cfg.stand_kd

        # initialize mujoco
        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.model.opt.timestep = self.cfg.simulation_dt
        self.data = mujoco.MjData(self.model)
        if self.cfg.visualization:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # init buffers
        self.command = np.zeros(self.cfg.num_actions)
        self.tau = np.zeros(self.cfg.num_actions)
        self._dof_pos = np.zeros(self.cfg.num_actions)
        self._dof_vel = np.zeros(self.cfg.num_actions)
        self.gamepad_cmd = np.zeros(3)
        self._base_proj_vec = np.zeros(3)
        self._base_ang_vel = np.zeros(3)
        self._base_lin_vel = np.zeros(3)
        self._base_quat = np.zeros(4)
        self._last_step_time = time.time()

        self.lag_buffer = [np.array(self.init_pos) for _ in range(self.cfg.lag_step)]

        self.logger = SimpleLogger()
        self.reset()

    def step(self):
        for _ in range(self.cfg.decimation):
            self.data.ctrl[:] = self.compute_torque()
            mujoco.mj_step(self.model, self.data)
            if self.cfg.visualization:
                if self.viewer.is_running():
                    with self.viewer.lock():
                        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)
                self.viewer.sync()
            self.log()

        time_until_next_step = self.model.opt.timestep * self.cfg.decimation - (time.time() - self._last_step_time)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        else:
            print("dt is too short!")

        self._last_step_time = time.time()

    def set_motor_command(self, kp, kd, command):
        self.kp = kp
        self.kd = kd
        self.command = self.lag_buffer[0]
        self.lag_buffer = self.lag_buffer[1:] + [command]

    def compute_torque(self):
        q = self.data.qpos[7:].copy()
        dq = self.data.qvel[6:].copy()
        self.tau = self.kp * (self.command - q) - self.kd * dq
        return self.tau

    def log(self):
        self.logger.log('dof_pos', self.dof_pos)
        self.logger.log('dof_vel', self.dof_vel)
        self.logger.log('base_quat', self.base_quat)
        self.logger.log('base_angle', R.from_quat(self.base_quat).as_euler('xyz', degrees=True))
        self.logger.log('base_ang_vel', self.base_ang_vel)
        self.logger.log('action', self.command)
        self.logger.log('dof_torque', self.dof_torque)
        self.logger.log('cmd', self.gamepad_cmd)

    def reset(self):
        self.data.qpos[2] = 0.10
        self.data.qpos[7:] = self.init_pos
        self.data.qvel[:] = np.zeros(18)
        for i in range(100):
            self.command = self.init_pos
            self.set_motor_command(self.cfg.stand_kp, self.cfg.stand_kd, self.command)
            self.step()

    def wake_up(self):
        self.data.qpos[0:3] = np.array([0, 20, 0.35])
        self.data.qpos[3:7] = np.array([1, 0, 0, 0])
        self.data.qpos[7:] = self.cfg.default_pos
        self.data.qvel[:] = np.zeros(24)

    @property
    def dof_pos(self) -> np.ndarray:
        self._dof_pos = self.data.qpos[7:].copy()
        return self._dof_pos.copy()

    @property
    def dof_vel(self) -> np.ndarray:
        self._dof_vel = self.data.qvel[6:].copy()
        return self._dof_vel.copy()

    @property
    def dof_torque(self) -> np.ndarray:
        return self.tau.copy()

    @property
    def base_quat(self) -> np.ndarray:
        quat = np.zeros(4)
        quat[0:3] = self.data.qpos[4:7]
        quat[3] = self.data.qpos[3]
        self._base_quat = quat
        return self._base_quat.copy()

    @property
    def base_ang_vel(self) -> np.ndarray:
        self._base_ang_vel = self.data.qvel[3:6].copy()
        return self._base_ang_vel.copy()

    @property
    def base_lin_vel(self) -> np.ndarray:
        rotation_mat_b2w = R.from_quat(self._base_quat).as_matrix()
        self._base_lin_vel = (rotation_mat_b2w.T.copy() @ self.data.qvel[0:3])
        return self._base_lin_vel.copy()

    @property
    def projected_gravity(self) -> np.ndarray:
        """return the projected gravity vector in world frame"""
        rot_mat = R.from_quat(self.base_quat).as_matrix()
        self._base_proj_vec = np.matmul(rot_mat.T, np.array([0, 0, -1]))
        return self._base_proj_vec.copy()


class Go2_real:
    def __init__(self, cfg: Config):
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
            from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_
            from unitree_sdk2py.utils.crc import CRC
            from unitree_sdk2py.utils.thread import Thread
            from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
            from unitree_sdk2py.go2.sport.sport_client import SportClient
        except:
            print("SDK Import Failed!")

        # check connection
        self.network_interface_card = cfg.network_interface_card
        # init sdk clients
        self._init_unitree_sdk()
        # Get Messages
        self.msg = None
        while not self.msg:
            time.sleep(0.01)

        self._foot_contact_logical_or = np.zeros(4)
        # shutdown sport mode
        status, result = self.motion_switcher_client.CheckMode()
        while result['name']:
            self.sport_client.StandDown()
            self.motion_switcher_client.ReleaseMode()
            status, result = self.motion_switcher_client.CheckMode()
            time.sleep(1)
        else:
            print("sport mode already shutdown")

        # init buffer
        self.cfg = cfg
        self.logger = SimpleLogger()
        self.gamepad_cmd = np.zeros(4)
        self.command = np.zeros(self.cfg.num_actions)
        self.kp = np.ones(self.cfg.num_actions) * self.cfg.kp
        self.kd = np.ones(self.cfg.num_actions) * self.cfg.kd

        self._last_step_time = time.time()
        self._base_quat = np.zeros(4)
        self._base_ang_vel = np.zeros(3)
        self._dof_pos = np.zeros(self.cfg.num_actions)
        self._dof_vel = np.zeros(self.cfg.num_actions)
        self._dof_torque = np.zeros(self.cfg.num_actions)
        self._projected_gravity = np.array([0, 0, 1])

        self._receive_observation()

    def step(self):
        self._send_dof_cmd(self.command)
        self._receive_observation()
        self.log()
        self._nap()

    def set_motor_command(self, kp, kd, command):
        self.kp = np.ones(self.cfg.num_actions) * kp
        self.kd = np.ones(self.cfg.num_actions) * kd
        self.command = command

    def _send_dof_cmd(self, tar_dof_pos):
        for motor_id, motor_angle in enumerate(tar_dof_pos):
            self.cmd.motor_cmd[motor_id].q = motor_angle  # Set to position(rad)
            self.cmd.motor_cmd[motor_id].kp = self.kp[motor_id]
            self.cmd.motor_cmd[motor_id].dq = 0.0  # Set to angular velocity(rad/s)
            self.cmd.motor_cmd[motor_id].kd = self.kd[motor_id]
            self.cmd.motor_cmd[motor_id].tau = 0.0  # target toque is set to xxx N.m

        self.cmd.crc = self.crc.Crc(self.cmd)

        # Publish message
        if not self.pub.Write(self.cmd):
            print("pub dof cmd error")

    def log(self):
        self.logger.log('dof_pos', self.dof_pos)
        self.logger.log('dof_vel', self.dof_vel)
        self.logger.log('base_quat', self.base_quat)
        self.logger.log('base_ang_vel', self.base_ang_vel)
        self.logger.log('action', self.command)
        self.logger.log('dof_torque', self.dof_torque)
        self.logger.log('cmd', self.gamepad_cmd)

    def _nap(self):
        """Sleep for the remainder of self.dt"""
        now = time.time()
        sleep_time = self.cfg.control_dt - (now - self._last_step_time)

        if sleep_time >= 0:
            time.sleep(sleep_time)
        else:
            print(f"dt is too short!{sleep_time}")

        self._last_step_time = time.time()

    def _init_unitree_sdk(self):
        try:
            ChannelFactoryInitialize(0, self.network_interface_card)

            self.sport_client = SportClient()
            self.sport_client.SetTimeout(5.0)
            self.sport_client.Init()

            self.motion_switcher_client = MotionSwitcherClient()
            self.motion_switcher_client.SetTimeout(5.0)
            self.motion_switcher_client.Init()

            self.crc = CRC()

            self.sub = ChannelSubscriber("rt/lowstate", LowState_)
            self.sub.Init(self.LowStateHandler, 10)

            self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
            self.pub.Init()

            self.cmd = unitree_go_msg_dds__LowCmd_()
            self.InitLowCmd()

        except Exception as e:
            print("sport_client or motion_switcher_client initialize failed!")

    def InitLowCmd(self):
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.cmd.motor_cmd[i].q = 2.146e9  # go2.PosStopF
            self.cmd.motor_cmd[i].kp = 0
            self.cmd.motor_cmd[i].dq = 16000.0  # VelStopF
            self.cmd.motor_cmd[i].kd = 0
            self.cmd.motor_cmd[i].tau = 0

    def LowStateHandler(self, msg):
        self.msg = msg

    def _receive_observation(self):
        # receive obs from robot sensor

        # NOTE: quaternion: wxyz -> xyzw
        q = self.msg.imu_state.quaternion
        self._base_quat = np.array([q[1], q[2], q[3], q[0]])
        if self._base_quat.sum == 0:
            warnings.warn("Robot not connected!")

        self.base_euler = np.array(self.msg.imu_state.rpy)
        self.base_euler_aligned = np.array([self.base_euler[0], self.base_euler[1], 0])
        self.rot_mat_aligned = R.from_euler('xyz', self.base_euler_aligned).as_matrix()

        self.base_gyro = np.array(self.msg.imu_state.gyroscope)  # rad/s
        self.base_acc = np.array(self.msg.imu_state.accelerometer)  # m/s^2

        # NOTE: dof in the order of [FR(hip thigh calf), FL, RR, RL]
        self._dof_pos = np.array([motor.q for motor in self.msg.motor_state[:12]])
        self._dof_vel = np.array([motor.dq for motor in self.msg.motor_state[:12]])
        self._dof_torque = np.array([motor.tau_est for motor in self.msg.motor_state[:12]])
        self._foot_force = np.array(self.msg.foot_force)

        contact_ = self._foot_force > 28.0
        self._foot_contact = np.where(np.logical_or(contact_, self._foot_contact_logical_or), 1.0, 0.0)
        self._foot_contact_logical_or = contact_.copy()

        ### calc  -----------
        self.rot_mat_aligned_world2base = np.linalg.inv(self.rot_mat_aligned)

        # self.base_ang_vel_body = self.rot_mat_aligned_world2base @ self.base_gyro
        self._base_ang_vel = self.base_gyro.copy()  ###  TODO

        self._projected_gravity = self.rot_mat_aligned_world2base @ np.array([0, 0, -1.0])

    @property
    def dof_pos(self) -> np.ndarray:
        return self._dof_pos.copy()

    @property
    def dof_vel(self) -> np.ndarray:
        return self._dof_vel.copy()

    @property
    def dof_torque(self) -> np.ndarray:
        return self._dof_torque.copy()

    @property
    def base_quat(self) -> np.ndarray:
        return self._base_quat.copy()

    @property
    def base_ang_vel(self) -> np.ndarray:
        return self._base_ang_vel.copy()

    @property
    def projected_gravity(self) -> np.ndarray:
        return self._projected_gravity.copy()

    @property
    def foot_contact_logical_or(self) -> np.ndarray:
        return self._foot_contact_logical_or.copy()
