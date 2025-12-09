import pygame
import threading
import time
import os
import platform
import sys

# 关闭GUI可视化（用于headless运行）
# os.environ["SDL_VIDEODRIVER"] = "dummy"


class Gamepad:
    def __init__(self, joystick_id=0, cfg=None):
        pygame.init()
        pygame.joystick.init()

        self._state = {
            'buttons': {},
            'axes': {},
            'hats': {},
            'connected': False
        }

        self._running = False
        self._thread = None
        self.lock = threading.Lock()
        self._joystick = None
        self._joystick_id = joystick_id

        # 参数设置
        self._vel_x_max = cfg["vel_x_max"] if cfg else 1.0
        self._vel_y_max = cfg["vel_y_max"] if cfg else 1.0
        self._vel_yaw_max = cfg["vel_yaw_max"] if cfg else 1.0

        # 状态初始化
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._vel_yaw = 0.0

        self._a_pressed = False
        self._b_pressed = False
        self._x_pressed = False
        self._y_pressed = False
        self._lb_pressed = False
        self._rb_pressed = False

        # 自动识别操作系统，设置手柄按钮映射
        system = platform.system()
        if system == "Darwin":  # macOS
            # ✅ 根据你测试的实际结果更新映射
            self._button_names = {
                0: 'a',  # A
                1: 'b',  # B
                2: 'x',  # X
                3: 'y',  # Y
                9: 'lb',  # LB
                10: 'rb',  # RB
                6: 'back',
                7: 'start',
                8: 'guide',
                11: 'ls',
                12: 'rs'
            }

            self._axis_names = {
                1: 'left_stick_y',  # 左摇杆Y轴（左右）
                0: 'left_stick_x',  # 左摇杆X轴（前后）
                2: 'right_stick_y',  # 右摇杆Y轴
            }

        elif system == "Windows":
            self._button_names = {
                0: 'a',
                1: 'b',
                2: 'x',
                3: 'y',
                4: 'lb',
                5: 'rb',
                6: 'back',
                7: 'start',
                8: 'guide',
                9: 'ls',
                10: 'rs'
            }

            self._axis_names = {
                0: 'left_stick_y',  # 左摇杆Y轴（左右）
                1: 'left_stick_x',  # 左摇杆X轴（前后）
                3: 'right_stick_y',  # 右摇杆Y轴
            }
        else:  # Linux 默认
            self._button_names = {
                0: 'a',
                1: 'b',
                2: 'x',
                3: 'y',
                4: 'lb',
                5: 'rb',
                6: 'back',
                7: 'start',
                8: 'guide',
                9: 'ls',
                10: 'rs'
            }

            self._axis_names = {
                1: 'left_stick_y',  # 左摇杆Y轴（左右）
                0: 'left_stick_x',  # 左摇杆X轴（前后）
                3: 'right_stick_y',  # 右摇杆Y轴
            }

        self._hat_names = {
            0: 'dpad'
        }

        for btn_id, btn_name in self._button_names.items():
            self._state['buttons'][btn_name] = False
        for axis_id, axis_name in self._axis_names.items():
            self._state['axes'][axis_name] = 0.0
        for hat_id, hat_name in self._hat_names.items():
            self._state['hats'][hat_name] = (0, 0)

        self._init_joystick()

    def _init_joystick(self):
        try:
            if pygame.joystick.get_count() > self._joystick_id:
                self._joystick = pygame.joystick.Joystick(self._joystick_id)
                self._joystick.init()
                self._state['connected'] = True
                print(f"✅ 手柄连接成功: {self._joystick.get_name()}")
            else:
                self._state['connected'] = False
                print("⚠️ 未检测到手柄连接")
        except Exception as e:
            print(f"❌ 手柄初始化失败: {e}")
            self._state['connected'] = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        print("🎮 手柄读取线程已启动 (100Hz)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print("🛑 手柄读取线程已停止")

    def _update_loop(self):
        update_interval = 0.01  # 100Hz
        while self._running:
            start_time = time.time()
            pygame.event.pump()
            self._update_state()
            self._update_velocities()
            elapsed = time.time() - start_time
            time.sleep(max(0, update_interval - elapsed))

    def _update_state(self):
        if not self._state['connected'] or not self._joystick:
            if pygame.joystick.get_count() > self._joystick_id:
                self._init_joystick()
            return

        try:
            with self.lock:
                for button_id in range(self._joystick.get_numbuttons()):
                    button_name = self._button_names.get(button_id, f'button_{button_id}')
                    self._state['buttons'][button_name] = bool(self._joystick.get_button(button_id))

                for axis_id, axis_name in self._axis_names.items():
                    if axis_id < self._joystick.get_numaxes():
                        self._state['axes'][axis_name] = self._joystick.get_axis(axis_id)

                for hat_id in range(self._joystick.get_numhats()):
                    hat_name = self._hat_names.get(hat_id, f'hat_{hat_id}')
                    self._state['hats'][hat_name] = self._joystick.get_hat(hat_id)

                self._a_pressed = self._state['buttons'].get('a', False)
                self._b_pressed = self._state['buttons'].get('b', False)
                self._x_pressed = self._state['buttons'].get('x', False)
                self._y_pressed = self._state['buttons'].get('y', False)
                self._lb_pressed = self._state['buttons'].get('lb', False)
                self._rb_pressed = self._state['buttons'].get('rb', False)

        except pygame.error as e:
            print(f"⚠️ 手柄读取错误: {e}")
            self._state['connected'] = False
            self._joystick = None

    def _update_velocities(self):
        with self.lock:
            self._vel_y = -self._state['axes'].get('left_stick_x', 0.0) * self._vel_y_max
            self._vel_x = -self._state['axes'].get('left_stick_y', 0.0) * self._vel_x_max
            self._vel_yaw = -self._state['axes'].get('right_stick_y', 0.0) * self._vel_yaw_max

    def debug_button_mapping(self):
        """调试模式：打印按钮编号，用于确定macOS下LB/RB位置"""
        if not self._joystick:
            print("❌ 未检测到手柄")
            return

        num_buttons = self._joystick.get_numbuttons()
        print(f"\n=== 按钮调试模式 (共 {num_buttons} 个按钮) ===")
        print("请依次按下 LB / RB / LT / RT / START / BACK / 其他按钮：\n")

        last_pressed = set()
        try:
            while True:
                pygame.event.pump()
                pressed_now = {i for i in range(num_buttons) if self._joystick.get_button(i)}
                newly_pressed = pressed_now - last_pressed
                if newly_pressed:
                    for i in newly_pressed:
                        print(f"按钮 {i} 被按下")
                last_pressed = pressed_now
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n退出按钮调试模式")

    # 添加所需的属性访问器
    @property
    def vel_x(self):
        with self.lock:
            return self._vel_x

    @property
    def vel_y(self):
        with self.lock:
            return self._vel_y

    @property
    def vel_yaw(self):
        with self.lock:
            return self._vel_yaw

    @property
    def a_pressed(self):
        with self.lock:
            return self._a_pressed

    @property
    def b_pressed(self):
        with self.lock:
            return self._b_pressed

    @property
    def x_pressed(self):
        with self.lock:
            return self._x_pressed

    @property
    def y_pressed(self):
        with self.lock:
            return self._y_pressed

    @property
    def lb_pressed(self):
        with self.lock:
            return self._lb_pressed

    @property
    def rb_pressed(self):
        with self.lock:
            return self._rb_pressed

    @property
    def dpad(self):
        with self.lock:
            return self._state['hats'].get('dpad', (0, 0))

    @property
    def connected(self):
        return self._state['connected']

    @property
    def state(self):
        with self.lock:
            return self._state.copy()

    def get_button(self, button_name):
        with self.lock:
            return self._state['buttons'].get(button_name, False)

    def get_axis(self, axis_name):
        with self.lock:
            return self._state['axes'].get(axis_name, 0.0)

    def get_hat(self, hat_name):
        with self.lock:
            return self._state['hats'].get(hat_name, (0, 0))

    def print_status(self):
        if not self.connected:
            return "手柄未连接"

        buttons = []
        for btn in ['a', 'b', 'x', 'y', 'lb', 'rb']:
            if getattr(self, f'{btn}_pressed'):
                buttons.append(btn.upper())
            else:
                buttons.append(btn)

        dpad_x, dpad_y = self.dpad

        status = (f"速度X：{self.vel_x:.3f} "
                  f"速度Y：{self.vel_y:.3f} "
                  f"速度Yaw：{self.vel_yaw:.3f} | "
                  f"左摇杆x：{self._state['axes'].get('left_stick_x', 0.0):.3f} "
                  f"左摇杆y：{self._state['axes'].get('left_stick_y', 0.0):.3f} | "
                  f"右摇杆y：{self._state['axes'].get('right_stick_y', 0.0):.3f} | "
                  f"按键：{', '.join(buttons)} | "
                  f"方向键：({dpad_x}, {dpad_y})")

        return status

    def __del__(self):
        self.stop()


if __name__ == "__main__":
    cfg = {"vel_x_max": 1, "vel_y_max": 1, "vel_yaw_max": 1}
    gamepad = Gamepad(cfg=cfg)

    # 如果传入 --debug 参数，则进入调试模式
    if "--debug" in sys.argv:
        gamepad.debug_button_mapping()
        sys.exit(0)

    gamepad.start()
    try:
        print("手柄测试开始，按下各个按钮测试响应，Ctrl+C退出")
        print("=" * 80)
        last_status = ""
        while True:
            current_status = gamepad.print_status()
            if current_status != last_status:
                print(current_status)
                last_status = current_status
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        gamepad.stop()
