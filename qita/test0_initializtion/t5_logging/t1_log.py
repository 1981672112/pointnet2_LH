import logging

'https://www.cnblogs.com/kangsf2017/p/14700833.html'
# 默认输出为warning级别
# 使用baseConfig()来指定日志输出级别
logging.basicConfig(level=logging.DEBUG, filename='demo.txt', filemode='w')  # DEBUG INFO WARNING...

"""
一.
    1 下面5个都会输出 不使用则输出下面3行
    2 日志输出到文件 没有文件会自己创建 默认是追加'a' 加模式filemode='w'
"""

# print('我在下面')  # 异步写入 并发 //用logging 就不要用print 否则控制台输出是混的 # pycharm开了run with python console 还是在上面
# logging.debug('This is a debug')
# logging.info('This is a info')
# logging.warning('This is a warning')
# logging.error('This is a error')
# logging.critical('This is a critical')

"""
二.
    1 输出一个变量行不行
"""
name = 'abc'
year = 13
logging.debug("姓名 %s 年龄 %d", name, year)
logging.debug("姓名 %s 年龄 %d" % (name, year))
logging.debug("姓名 {} 年龄 {}".format(name, year))
logging.debug(f"姓名 {name} 年龄 {year}")
