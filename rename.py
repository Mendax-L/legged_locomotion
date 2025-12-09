import os

# 定义需要查找和替换的字符串
old_names = ['legged_locomotion', 'legged_locomotion']
new_name = 'legged_locomotion'

# 获取当前目录
directory = '/home/luxiao/legged_locomotion'  # 替换为你的目标文件夹路径

# 遍历文件夹内的所有内容
for root, dirs, files in os.walk(directory, topdown=False):
    # 重命名文件夹
    for name in dirs:
        # 在文件夹名称中查找并替换
        if any(old_name in name for old_name in old_names):
            old_folder_path = os.path.join(root, name)
            new_folder_path = os.path.join(root, name)
            # 对名称中的每个旧字符串进行替换
            for old_name in old_names:
                new_folder_path = new_folder_path.replace(old_name, new_name)
            
            if old_folder_path != new_folder_path:
                print(f'Renaming folder: {old_folder_path} -> {new_folder_path}')
                os.rename(old_folder_path, new_folder_path)

    # 重命名文件
    for name in files:
        # 在文件名中查找并替换
        for old_name in old_names:
            if old_name in name:
                old_file_path = os.path.join(root, name)
                new_file_path = os.path.join(root, name.replace(old_name, new_name))
                print(f'Renaming file: {old_file_path} -> {new_file_path}')
                os.rename(old_file_path, new_file_path)
