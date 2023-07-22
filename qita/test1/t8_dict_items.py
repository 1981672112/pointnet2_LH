dict = {'老大': '15岁',
        '老二': '14岁',
        '老三': '2岁',
        '老四': '在墙上',
        '老无': 5,
        }
print(dict.items())

# 2个接受 键和值分别返回
for key, values in dict.items():
    print(key + '已经' + str(values) + '了')
"""
老大已经15岁了
老二已经14岁了
老三已经2岁了
老四已经在墙上了
老无已经5了
"""

# 1个接受 键和值组成一个元组返回
for i in dict.items():
    print(i)
"""
('老大', '15岁')
('老二', '14岁')
('老三', '2岁')
('老四', '在墙上')
('老无', 5)
"""