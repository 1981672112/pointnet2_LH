# 面向对象 C++ java C# python   继承/封装/多态
# 设计模式：面向对象 (工厂模式等)

# 网页
# 爬虫java和python 但是python资料多 3个最基本的库

import requests
# 调用浏览器的插件
import selenium
# 企业级爬虫 分布式爬虫
import scrapy

# 两个网站开发的库
# import flask
# import django

import time


def decorator(fun):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        re = fun(*args, **kwargs)
        t2 = time.time()
        cost_time = t2 - t1
        return cost_time, re

    return wrapper


@decorator
def calculate(x):
    s = 0
    for i in range(x):
        s += i
    return s


hh = calculate(100)
print(hh[0], hh[1])
t = calculate(100)
print(t)
