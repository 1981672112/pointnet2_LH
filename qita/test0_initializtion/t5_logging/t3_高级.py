import logging

# 1
# 记录器
logger = logging.getLogger('applog')
logger.setLevel(logging.INFO)

# 处理器
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)

# 没有给handler指定日志级别，将使用logger的级别
fileHandler = logging.FileHandler(filename='addDemo.log')

# formatter格式
formatter = logging.Formatter("%(asctime)s|%(levelname)8s|%(filename)10s%lineno)s|%(message)s")

# 给处理器设置格式
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

# 记录器要设置处理器
logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

# 定义一个过滤器
flt = logging.Filter("cn.cccb")


# 关联过滤器
# logger.addFilter(flt)
fileHandler.addFilter(flt)

# 打印日志的代码
# logging.debug() #不能使用这个了！！！会使用WARNING的版本，不会用之前的记录器 用logger.debug
name = 'abc'
age = 13
logger.debug("姓名 %s, 年龄%d", name, age)
logger.debug("姓名 %s, 年龄%d",  (name, age))
logger.debug("姓名 {}, 年龄{}".format(name, age))
logger.debug(f"姓名{name}, 年龄{age}")
