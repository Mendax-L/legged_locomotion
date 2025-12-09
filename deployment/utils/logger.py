"""
@Author: Yang Xuekang
@E-mail: yangxuekang@sjtu.edu.cn
@Date  : 2025/11/3
"""


class SimpleLogger(dict):
    def __init__(self) -> None:
        super().__init__()

    def log(self, key, value):
        if key not in self:
            self[key] = []

        self[key].append(value)

    def get(self, key):
        return self[key]

    def clear(self):
        self.clear()
