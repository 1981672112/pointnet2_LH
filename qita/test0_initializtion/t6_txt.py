import random

# 读取原始txt文件
with open('/home/lh/兔子/bunny.txt', 'r') as file:
    lines = file.readlines()

# 随机选择一定数量的行
num_lines_to_select = 15000  # 选择10行，你可以根据需要更改这个数字
selected_lines = random.sample(lines, num_lines_to_select)

# 写入新的txt文件
with open('新文件.txt', 'w') as new_file:
    new_file.writelines(selected_lines)