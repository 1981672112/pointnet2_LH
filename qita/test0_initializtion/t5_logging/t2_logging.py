import logging

"""
输出格式和一些公共信息
"""
logging.basicConfig(format="%(asctime)s||%(levelname)s:%(lineno)s||%(message)s", level=logging.DEBUG, datefmt='%Y:%m:%d %H:%M,%Sms')
# print(logging.DEBUG)  # 10
# print(logging.WARNING)  # 30
name = 'abc'
year = 13
logging.debug("姓名 %s 年龄 %d", name, year)
logging.warning("姓名 %s 年龄 %d", name, year)
