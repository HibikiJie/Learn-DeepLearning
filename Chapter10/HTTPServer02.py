from flask import Flask,request
from PIL import Image
import io
app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World"


@app.route('/xxx', methods=['POST'])
def xxx():
    massage = request.form.get("name")
    file = request.files.get('file')
    image_byte = file.read()
    image = Image.open(io.BytesIO(image_byte))
    image.show()
    print(massage)
    return "收到来自客户端的消息："+massage


if __name__ == '__main__':
    """启动服务,可以通过port指定端口号"""
    app.run(port=5005)
