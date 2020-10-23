from flask import Flask,request


app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World"


@app.route('/xxx', methods=['POST'])
def xxx():
    # massage = request.args.get("name")
    massage = request.form.get("name")
    print(massage)
    return "收到来自客户端的消息："+massage


if __name__ == '__main__':
    """启动服务,可以通过port指定端口号"""
    app.run(port=5005)
