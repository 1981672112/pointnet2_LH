import datetime

import time

"""
1. 首先返回系统时间
2. 返回当天日期
3. 时间间隔（这是一个time模块很有用的）
"""
dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')  # 含微秒的日期时间，来源 比特量化
print(dt)  # 2023-03-30 20:33:22
print(dt_ms)  # 2023-03-30 20:33:22.652360
nowTime = datetime.datetime.now()
print(nowTime)  # 2023-03-30 20:33:22.652360

Today = datetime.date.today()  # 只显示年月日
print(Today)  # 2023-03-30


def sleeptime(hour, min, sec):
    return hour * 3600 + min * 60 + sec  # 计算一个 时间段（长度）


sleep_time = sleeptime(0, 0, 5)  # sleep_time=时间段（长度）

while 1 == 1:
    time.sleep(sleep_time)
    print("每隔5秒显示一次")
