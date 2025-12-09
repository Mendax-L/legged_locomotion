import numpy as np


class Config:
    is_real = False  # FIXME real robot or not, See Args.
    use_gamepad = True
    fixed_command = False  # True 则不使用手柄，使用下面的固定命令 False 使用手柄
    command = [0.0, -0, 0, 0]
    model_path = "/home/luxiao/Go2_locomotion/logs/rsl_rl/unitree_go2_flat/2025-11-19_19-44-38/exported/policy.pt"  # flat
    xml_path = "/home/luxiao/Go2_locomotion/deployment/resources/go2/scene.xml"

    # mujoco simulation
    visualization = True
    simulation_dt = 0.001
    decimation = 20
    control_dt = simulation_dt * decimation

    # kp & kd
    stand_kp = 100  # stand up
    stand_kd = 5
    kp = 25  # walking
    kd = 0.5

    lag_step = 1  # 1 is no lag

    # change index
    sim2real = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
    real2sim = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]  # REAL: [FR, FL, RR, RL]

    # default pos
    default_pos = np.array([
        # FL FR RL RR 注意这里是IsaacLab的顺序，左始
        0.00, 0.00, 0.00, 0.00,  # Hip
        0.80, 0.80, 0.80, 0.80,  # Thigh
        -1.5, -1.5, -1.5, -1.5,  # Calf
    ])

    #  init pos
    init_pos = np.array([
        # FL FR RL RR 注意这里是IsaacLab的顺序，左始
        0.13, -0.13, 0.50, -0.50,  # Hip
        1.22, 1.22, 1.22, 1.22,  # Thigh
        -2.72, -2.72, -2.72, -2.72,  # Calf
    ])

    # obs num
    num_obs = 45  # 注意obs之间的顺序
    num_critic_obs = 3 + 3  # com_displacement lin_vel, only for init
    num_actions = 12

    # action scale
    action_scale = 0.25

    # action clip
    action_clip = 5

    # REAL Go2 Config
    network_interface_card = "enp52s0"  # name of network_interface_card that connecting to Go2
