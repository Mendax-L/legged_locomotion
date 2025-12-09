import threading
import time
import ctypes
from ctypes import wintypes

class Gamepad:
    """基于 Windows Input API 的手柄驱动，接口与 pygame 版本一致"""
    
    # Windows API 常量
    XINPUT_GAMEPAD_DPAD_UP = 0x0001
    XINPUT_GAMEPAD_DPAD_DOWN = 0x0002
    XINPUT_GAMEPAD_DPAD_LEFT = 0x0004
    XINPUT_GAMEPAD_DPAD_RIGHT = 0x0008
    XINPUT_GAMEPAD_START = 0x0010
    XINPUT_GAMEPAD_BACK = 0x0020
    XINPUT_GAMEPAD_LEFT_THUMB = 0x0040
    XINPUT_GAMEPAD_RIGHT_THUMB = 0x0080
    XINPUT_GAMEPAD_LB = 0x0100
    XINPUT_GAMEPAD_RB = 0x0200
    XINPUT_GAMEPAD_A = 0x1000
    XINPUT_GAMEPAD_B = 0x2000
    XINPUT_GAMEPAD_X = 0x4000
    XINPUT_GAMEPAD_Y = 0x8000

    def __init__(self, joystick_id=0, cfg=None):
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

        self._state = {
            'buttons': {},
            'axes': {},
            'hats': {},
            'connected': False
        }

        # 按钮映射
        self._button_names = {
            'a': self.XINPUT_GAMEPAD_A,
            'b': self.XINPUT_GAMEPAD_B,
            'x': self.XINPUT_GAMEPAD_X,
            'y': self.XINPUT_GAMEPAD_Y,
            'lb': self.XINPUT_GAMEPAD_LB,
            'rb': self.XINPUT_GAMEPAD_RB,
            'back': self.XINPUT_GAMEPAD_BACK,
            'start': self.XINPUT_GAMEPAD_START,
            'ls': self.XINPUT_GAMEPAD_LEFT_THUMB,
            'rs': self.XINPUT_GAMEPAD_RIGHT_THUMB,
        }

        self._axis_names = {
            'left_stick_x': 0,
            'left_stick_y': 1,
            'right_stick_y': 3,
        }

        self._hat_names = {
            'dpad': 0
        }

        # 初始化状态字典
        for btn_name in self._button_names.keys():
            self._state['buttons'][btn_name] = False
        for axis_name in self._axis_names.keys():
            self._state['axes'][axis_name] = 0.0
        self._state['hats']['dpad'] = (0, 0)

        # 线程控制
        self._running = False
        self._thread = None
        self.lock = threading.Lock()

        # 加载 XInput DLL
        try:
            self._xinput = ctypes.windll.xinput1_4
            print(f"✅ XInput 1.4 加载成功")
        except OSError:
            try:
                self._xinput = ctypes.windll.xinput1_3
                print(f"✅ XInput 1.3 加载成功")
            except OSError:
                self._xinput = ctypes.windll.xinput9_1_0
                print(f"✅ XInput 9.1.0 加载成功")

        self._init_joystick()

    def _init_joystick(self):
        try:
            self._state['connected'] = self._check_connection()
            if self._state['connected']:
                print(f"✅ 手柄连接成功 (UserIndex: {self._joystick_id})")
            else:
                print("⚠️ 未检测到手柄连接")
        except Exception as e:
            print(f"❌ 手柄初始化失败: {e}")
            self._state['connected'] = False

    def _check_connection(self):
        """检查手柄是否连接"""
        try:
            result = self._xinput.XInputGetState(
                ctypes.c_uint(self._joystick_id),
                ctypes.byref(ctypes.c_char(b'\x00' * 16))
            )
            return result == 0
        except:
            return False

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
            self._update_state()
            self._update_velocities()
            elapsed = time.time() - start_time
            time.sleep(max(0, update_interval - elapsed))

    def _update_state(self):
        """从 XInput 读取手柄状态"""
        try:
            # 定义 XINPUT_STATE 结构体
            class XINPUT_GAMEPAD(ctypes.Structure):
                _fields_ = [
                    ("wButtons", wintypes.WORD),
                    ("bLeftTrigger", ctypes.c_ubyte),
                    ("bRightTrigger", ctypes.c_ubyte),
                    ("sThumbLX", wintypes.SHORT),
                    ("sThumbLY", wintypes.SHORT),
                    ("sThumbRX", wintypes.SHORT),
                    ("sThumbRY", wintypes.SHORT),
                ]

            class XINPUT_STATE(ctypes.Structure):
                _fields_ = [
                    ("dwPacketNumber", wintypes.DWORD),
                    ("Gamepad", XINPUT_GAMEPAD),
                ]

            state = XINPUT_STATE()
            result = self._xinput.XInputGetState(
                ctypes.c_uint(self._joystick_id),
                ctypes.byref(state)
            )

            with self.lock:
                if result == 0:  # ERROR_SUCCESS
                    self._state['connected'] = True
                    buttons = state.Gamepad.wButtons

                    # 更新按钮状态
                    self._state['buttons']['a'] = bool(buttons & self.XINPUT_GAMEPAD_A)
                    self._state['buttons']['b'] = bool(buttons & self.XINPUT_GAMEPAD_B)
                    self._state['buttons']['x'] = bool(buttons & self.XINPUT_GAMEPAD_X)
                    self._state['buttons']['y'] = bool(buttons & self.XINPUT_GAMEPAD_Y)
                    self._state['buttons']['lb'] = bool(buttons & self.XINPUT_GAMEPAD_LB)
                    self._state['buttons']['rb'] = bool(buttons & self.XINPUT_GAMEPAD_RB)
                    self._state['buttons']['back'] = bool(buttons & self.XINPUT_GAMEPAD_BACK)
                    self._state['buttons']['start'] = bool(buttons & self.XINPUT_GAMEPAD_START)
                    self._state['buttons']['ls'] = bool(buttons & self.XINPUT_GAMEPAD_LEFT_THUMB)
                    self._state['buttons']['rs'] = bool(buttons & self.XINPUT_GAMEPAD_RIGHT_THUMB)

                    # 更新轴状态（标准化到 -1.0 ~ 1.0）
                    self._state['axes']['left_stick_x'] = state.Gamepad.sThumbLX / 32768.0
                    self._state['axes']['left_stick_y'] = state.Gamepad.sThumbLY / 32768.0
                    self._state['axes']['right_stick_y'] = state.Gamepad.sThumbRY / 32768.0

                    # 更新方向键
                    dpad_x = 0
                    dpad_y = 0
                    if buttons & self.XINPUT_GAMEPAD_DPAD_LEFT:
                        dpad_x = -1
                    elif buttons & self.XINPUT_GAMEPAD_DPAD_RIGHT:
                        dpad_x = 1
                    if buttons & self.XINPUT_GAMEPAD_DPAD_UP:
                        dpad_y = 1
                    elif buttons & self.XINPUT_GAMEPAD_DPAD_DOWN:
                        dpad_y = -1
                    self._state['hats']['dpad'] = (dpad_x, dpad_y)

                    # 更新按键快捷属性
                    self._a_pressed = self._state['buttons']['a']
                    self._b_pressed = self._state['buttons']['b']
                    self._x_pressed = self._state['buttons']['x']
                    self._y_pressed = self._state['buttons']['y']
                    self._lb_pressed = self._state['buttons']['lb']
                    self._rb_pressed = self._state['buttons']['rb']
                else:
                    self._state['connected'] = False

        except Exception as e:
            print(f"⚠️ 手柄读取错误: {e}")
            with self.lock:
                self._state['connected'] = False

    def _update_velocities(self):
        with self.lock:
            self._vel_y = -self._state['axes'].get('left_stick_x', 0.0) * self._vel_y_max
            self._vel_x = -self._state['axes'].get('left_stick_y', 0.0) * self._vel_x_max
            self._vel_yaw = -self._state['axes'].get('right_stick_y', 0.0) * self._vel_yaw_max

    # 属性访问器（与 pygame 版本一致）
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

    def debug_button_mapping(self):
        """调试模式"""
        if not self.connected:
            print("❌ 未检测到手柄")
            return

        print("\n=== 按钮调试模式 ===")
        print("请依次按下各个按钮，Ctrl+C退出：\n")

        last_pressed = set()
        try:
            while True:
                current = set()
                for btn_name, btn_value in self._button_names.items():
                    if self.get_button(btn_name):
                        current.add(btn_name)
                
                newly_pressed = current - last_pressed
                if newly_pressed:
                    for btn in newly_pressed:
                        print(f"按钮 {btn} 被按下")
                last_pressed = current
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n退出按钮调试模式")

    def __del__(self):
        self.stop()


if __name__ == "__main__":
    cfg = {"vel_x_max": 1, "vel_y_max": 1, "vel_yaw_max": 1}
    gamepad = Gamepad(cfg=cfg)
    gamepad.start()
    try:
        print("手柄测试开始，Ctrl+C退出")
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