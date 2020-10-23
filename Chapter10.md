# Chapter10

# 工程部署

部署内容，全在[pytorch](https://pytorch.org/)官网上。

只需要打包为.pt文件



1、通过带有FLASK的REST API在PYTHON中部署PYTORCH

2、[移动端](https://pytorch.org/mobile/android/)将'model.pt'文件和该文件给软件开发人员，还有输入输出格式。

# 1、HTTP部署

pytorch官网有详细教程，[rest](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html#)

由表单接受数据，json发送数据。

![image-20201017154500214](D:%5CLearn-DeepLearning%5Cimage%5Cimage-20201017154500214.png)

由客服端，请求服务器。首先需要知道地址：

cmd中输入命令：

```python
ipconfig
```

返回：

```
Windows IP 配置
以太网适配器 以太网:
   连接特定的 DNS 后缀 . . . . . . . :
   本地链接 IPv6 地址. . . . . . . . : fe80::247b:6adb:ea18:297b%10
   IPv4 地址 . . . . . . . . . . . . : 192.168.1.7
   子网掩码  . . . . . . . . . . . . : 255.255.255.0
   默认网关. . . . . . . . . . . . . : 192.168.1.253
```

得到地址。

也可以“ping 地址“，来看能否访问



返回值：400以上，未找到

​				500或是以上，数据处理异常

​				200或以上，成功了



## （1）构建服务器

```python
from flask import Flask,request
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World"

@app.route('/xxx')
def xxx():
    massage = request.args.get("name")
    print(massage)
    return "xxxxxxx"

if __name__ == '__main__':
    """启动服务,可以通过port指定端口号"""
    app.run(port=5005)
```

其中，客户端，可以通过：

http://127.0.0.1:5005/xxx?name=ok中的？name = ok

给服务端发送一个ok的信息

服务器通过request接受

```python
massage = request.args.get("name")
```

这是一个get请求，通过地址栏发送请求。这种方式，不安全。速度快，可以传递简短的字符信息。



POST表单形式

html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>显示页面</title>
</head>
<body>
    <form action="http://127.0.0.1:5005/xxx" method="post">
        <input name="name" value="">
        <input type="submit" value="提交">
    </form>
</body>
</html>
```



传输文件，

```html
<form action="http://127.0.0.1:5005/xxx" method="post" enctype="multipart/form-data">
    <input type="file" name="file">
    <input name="name" value="">
    <input type="submit" value="提交">
</form>
```

服务端通过`file = request.files.get('file')`

获取文件，

在通过：

```python
image_byte = file.read()
image = Image.open(io.BytesIO(image_byte))
```

将二进制流转换为文件。



python socket通讯

# 2、ONNX

（1）导出为“.onnx”文件

安装onnx库，

```python
pip install onnx onnxruntime
```

```python
"""创建输入张量、输出张量"""
batch_size = 1
W = 640
H = 480
x_p = torch.randn(batch_size, 3, H, W, requires_grad=True)
torch_out_p = p_net(x_p)

"""导出模型"""
"""导出P网络的模型，设置动态轴，否则输入尺寸将在导出的ONNX图形中固定为所有输入尺寸。"""
dynamics_axes1 = {
    'input': {2: 'H', 3: 'W'},
    'output': {2: 'H', 3: 'W'}
}
torch.onnx.export(p_net,
                  x_p,
                  "mtcnn_onnx/p_net.onnx",
                  export_params=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes=dynamics_axes1
                  )
```

检查onnx模型的有效性。将验证模型的结构并确认模型具有有效的架构。通过检查模型的版本，图形的结构以及节点及其输入和输出，可以验证ONNX图的有效性。

```python

import onnx

onnx_model_p = onnx.load("mtcnn_onnx/p_net.onnx")
onnx.checker.check_model(onnx_model_p)
```



使用ONNX Runtime运行模型；验证数值计算是否相同

```python
import onnxruntime


def check(x, torch_out, onnx):
    ort_session = onnxruntime.InferenceSession(onnx)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outs[1], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
check(x_p,torch_out_p,"mtcnn_onnx/p_net.onnx")
```

于是打包完成。



（2）使用onnx文件

加载文件，

```python
import onnxruntime
ort_session = onnxruntime.InferenceSession("p_net.onnx")
```



创建输入，然后调用run函数，即可进行一次推理

```python
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
ort_outs = ort_session.run(None, ort_inputs)
```





# 3、将onnx的mtcnn布置与HTTP上

提前将侦测网络加载进服务器，但客服端提交过来图片，输出人脸框。

```python
explorer = Explorer()
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
```