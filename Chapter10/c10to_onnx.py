from torch import nn
import torch
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch.onnx

"""创建MTCNN的三个模型"""


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.PReLU(),

            nn.Conv2d(10, 16, 3, 1),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.PReLU()
        )
        self.classification = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        self.regression = nn.Conv2d(32, 4, 1)

    def forward(self, enter):
        enter = self.conv1(enter)
        return self.classification(enter), self.regression(enter)


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.PReLU(),

            nn.Conv2d(28, 48, 3, 1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.PReLU(),

            nn.Conv2d(48, 64, 2, 1),
            nn.PReLU()
        )
        self.full_connect = nn.Sequential(
            nn.Linear(3 * 3 * 64, 128),
            nn.PReLU()
        )
        self.classification = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.regression = nn.Linear(128, 4)

    def forward(self, enter):
        enter = self.full_connect(self.convolution(enter).reshape(-1, 576))
        return self.classification(enter), self.regression(enter)


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.PReLU(),

            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(3, 2),
            nn.PReLU(),

            nn.Conv2d(64, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.PReLU(),

            nn.Conv2d(64, 128, 2, 1),
            nn.PReLU()
        )
        self.fully_connect = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU()
        )
        self.classification = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.regression = nn.Linear(256, 4)

    def forward(self, enter):
        enter = self.fully_connect(self.convolution(enter).reshape(-1, 1152))
        return self.classification(enter), self.regression(enter)


"""创建模型，加载参数，并切换为推理模式"""
p_net = PNet().eval()
p_net.load_state_dict(torch.load('D:/Learn-DeepLearning/project1/MTCNNCUDA/pnet.pth'))
r_net = RNet().eval()
r_net.load_state_dict(torch.load('D:/Learn-DeepLearning/project1/MTCNNCUDA/rnet.pth'))
o_net = ONet().eval()
o_net.load_state_dict(torch.load('D:/Learn-DeepLearning/project1/MTCNNCUDA/onet.pth'))

"""创建输入张量"""
batch_size = 1
W = 640
H = 480
x_p = torch.randn(batch_size, 3, H, W, requires_grad=True)
x_r = torch.randn(batch_size, 3, 24, 24, requires_grad=True)
x_o = torch.randn(batch_size, 3, 48, 48, requires_grad=True)
torch_out_p = p_net(x_p)
torch_out_r = r_net(x_r)
torch_out_o = o_net(x_o)

"""导出模型"""
"""导出P网络的模型"""
dynamics_axes1 = {
    'input': {2: 'H', 3: 'W'},
    'output': {2: 'H', 3: 'W'}
}
dynamics_axes2 = {
    'input': {0: 'batch_size'},
    'output': {0: 'batch_size'}
}
torch.onnx.export(p_net,
                  x_p,
                  "mtcnn_onnx/p_net.onnx",
                  export_params=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes=dynamics_axes1
                  )

torch.onnx.export(r_net,
                  x_r,
                  "mtcnn_onnx/r_net.onnx",
                  export_params=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes=dynamics_axes2
                  )

torch.onnx.export(o_net,
                  x_o,
                  "mtcnn_onnx/o_net.onnx",
                  export_params=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes=dynamics_axes2
                  )

"""检查onnx模型的有效性。
将验证模型的结构并确认模型具有有效的架构。
通过检查模型的版本，图形的结构以及节点及其输入和输出，
可以验证ONNX图的有效性。"""
import onnx

onnx_model_p = onnx.load("mtcnn_onnx/p_net.onnx")
onnx.checker.check_model(onnx_model_p)
onnx_model_r = onnx.load("mtcnn_onnx/r_net.onnx")
onnx.checker.check_model(onnx_model_r)
onnx_model_o = onnx.load("mtcnn_onnx/o_net.onnx")
onnx.checker.check_model(onnx_model_o)


"""使用ONNX Runtime运行模型；验证数值计算是否相同"""
import onnxruntime


def check(x, torch_out, onnx):
    ort_session = onnxruntime.InferenceSession(onnx)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {'input': to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outs[1], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
check(x_p,torch_out_p,"mtcnn_onnx/p_net.onnx")
check(x_r,torch_out_r,"mtcnn_onnx/r_net.onnx")
check(x_o,torch_out_o,"mtcnn_onnx/o_net.onnx")

