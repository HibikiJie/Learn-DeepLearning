from flask import Flask,request,jsonify
from PIL import Image
from Chapter10.c10use_onnx_mtcnn import Explorer
import io
import numpy
app = Flask(__name__)
explorer = Explorer()

@app.route('/')
def hello():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    image_byte = file.read()
    image = Image.open(io.BytesIO(image_byte))
    image = numpy.array(image)
    boxes = explorer.explore(image)
    persons = {}
    for i,box in enumerate(boxes):
        persons[f'{i}'] = box.astype(numpy.int).tolist()
    return jsonify(persons)


if __name__ == '__main__':
    """启动服务,可以通过port指定端口号"""
    app.run(port=5000)