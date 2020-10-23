from flask import Flask,request


app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World"


@app.route('/xxx', methods=['POST', 'GET'])
def xxx():
    massage = request.args.get("name")
    print(massage)
    return "xxxxxxx"


if __name__ == '__main__':
    """启动服务,可以通过port指定端口号"""
    app.run(port=5005)
