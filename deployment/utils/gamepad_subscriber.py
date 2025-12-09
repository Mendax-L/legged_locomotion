"""
@Author: Yang Xuekang
@E-mail: yangxuekang@sjtu.edu.cn
@Date  : 2025/5/18
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import threading


class GamepadSubscriber(Node):
    def __init__(self, topic_name='/joy', cfg=None):
        super().__init__('joystick_reader')

        self.cfg = cfg

        self._vel_x = 0.0
        self._vel_y = 0.0
        self._vel_yaw = 0.0

        self._vel_x_max = self.cfg.vel_x_max if cfg is not None else 1
        self._vel_y_max = self.cfg.vel_y_max if cfg is not None else 1
        self._vel_yaw_max = self.cfg.vel_yaw_max if cfg is not None else 1

        self._a_pressed = False
        self._b_pressed = False
        self._x_pressed = False
        self._y_pressed = False

        self.lock = threading.Lock()

        self.subscription = self.create_subscription(Joy, topic_name, self.joy_callback, 10)

        self.connected = True
        self.update_thread = threading.Thread(target=self.run, daemon=True)
        self.update_thread.start()

    def joy_callback(self, msg: Joy):
        with self.lock:
            try:
                self._vel_x = msg.axes[1] * self._vel_x_max
                self._vel_y = msg.axes[0] * self._vel_y_max
                self._vel_yaw = msg.axes[3] * self._vel_yaw_max

                self._a_pressed = bool(msg.buttons[0])
                self._b_pressed = bool(msg.buttons[1])
                self._x_pressed = bool(msg.buttons[2])
                self._y_pressed = bool(msg.buttons[3])
            except IndexError as e:
                self.get_logger().warn(f"Joystick input index error: {e}")

    def run(self):
        while rclpy.ok() and self.connected:
            try:
                rclpy.spin_once(self, timeout_sec=0.005)
            except:
                pass

    def stop(self):
        self.connected = False
        self.update_thread.join()
        self.destroy_node()

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


if __name__ == '__main__':
    rclpy.init()
    joystick = GamepadSubscriber()

    try:
        while True:
            print(f"vx={joystick.vel_x:.2f}, vy={joystick.vel_y:.2f}, yaw={joystick.vel_yaw:.2f}")
            print(f"A={joystick.a_pressed}, B={joystick.b_pressed}, X={joystick.x_pressed}, Y={joystick.y_pressed}")
    except KeyboardInterrupt:
        pass
    finally:
        joystick.stop()
        rclpy.shutdown()
