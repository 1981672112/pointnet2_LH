# __all__ = ['User']  # 只针对from 包.模块 import * 起作用

version = "1.1"


class User():
    def __init__(self, username, password):
        self.username = username
        self.password = password

    def login(self, username, password):
        if username == self.username and password == self.password:
            print("登录成功")
        else:
            print("登录失败")

    def publish_article(self, article):
        print(self.username, "发表了文章", article.name)

    def show(self):
        print(self.username, self.password)
