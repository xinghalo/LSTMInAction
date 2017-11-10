# -*- coding: utf-8 -*-
import sys
reload(sys)  # Python2.7 初始化后会删除 sys.setdefaultencoding 这个方法，我们需要重新载入
sys.setdefaultencoding("utf-8")

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    # 默认是127.0.0.1:5000
    app.run()