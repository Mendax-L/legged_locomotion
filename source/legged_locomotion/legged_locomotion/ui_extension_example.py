# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.ext

# Functions and vars are available to other extension as usual in python: `example.python_ext.some_public_function(x)`
# 函数和变量可以像普通 Python 模块一样被其他扩展访问：例如 `example.python_ext.some_public_function(x)`
def some_public_function(x: int):
    # A small example public function that prints a message and returns x**x.
    # 一个简单的示例公共函数：打印信息并返回 x 的 x 次幂。
    # This demonstrates how other extensions can import and use functions from this module.
    # 该函数用于演示其他扩展如何导入并使用此模块中的函数。
    print("[Go2_locomotion] some_public_function was called with x: ", x)
    return x**x


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
# 任何继承自 `omni.ext.IExt` 的顶级模块类（在 extension.toml 的 `python.modules` 中声明）将在扩展启用时被实例化，
# 并调用其 on_startup(ext_id) 方法；当扩展被禁用时会调用 on_shutdown()。
class ExampleExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    # ext_id 是当前扩展的标识符。可以配合扩展管理器查询更多信息，例如扩展在文件系统中的位置等。
    def on_startup(self, ext_id):
        # Called by the Omniverse when the extension is enabled.
        # 当扩展被启用时由 Omniverse 调用（初始化入口）。
        print("[Go2_locomotion] startup")

        # Simple internal counter to demonstrate stateful UI updates.
        # 简单内部计数器，用于示例化有状态的 UI 更新。
        self._count = 0

        # Create a simple window with a label and two buttons using omni.ui.
        # 使用 omni.ui 创建一个简单窗口，包含标签和两个按钮。
        self._window = omni.ui.Window("My Window", width=300, height=300)
        # The UI is built inside the window frame context so it is properly parented and cleaned up.
        # UI 在窗口帧上下文中创建，保证父子关系正确并在窗口关闭时被清理。
        with self._window.frame:
            with omni.ui.VStack():
                # Label to display the counter value.
                # 用于显示计数值的标签控件。
                label = omni.ui.Label("")

                def on_click():
                    # Increment the counter and update the label text.
                    # 点击 "Add" 时增加计数并更新标签显示。
                    self._count += 1
                    label.text = f"count: {self._count}"

                def on_reset():
                    # Reset the counter and clear the label.
                    # 点击 "Reset" 时重置计数并清空标签。
                    self._count = 0
                    label.text = "empty"

                # Initialize the UI state by calling on_reset once.
                # 通过调用 on_reset 初始化 UI 状态（确保标签有初始文本）。
                on_reset()

                # Place two buttons horizontally: Add and Reset.
                # 使用水平布局放置两个按钮：Add（增加计数）和 Reset（重置计数）。
                with omni.ui.HStack():
                    omni.ui.Button("Add", clicked_fn=on_click)
                    omni.ui.Button("Reset", clicked_fn=on_reset)

    def on_shutdown(self):
        # Called by the Omniverse when the extension is disabled.
        # 当扩展被禁用时由 Omniverse 调用（清理入口）。
        # If you create persistent resources (threads, handlers, window references), clean them up here.
        # 如果创建了持久化资源（线程、回调处理器、窗口引用等），应在此处释放/注销。
        print("[Go2_locomotion] shutdown")