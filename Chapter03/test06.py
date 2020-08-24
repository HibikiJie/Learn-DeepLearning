import torch
import numpy

class MyArcCos(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.acos(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = -(1-input**2)**(-0.5)
        grad_input[input >= 0.866025403784438646] = -2*input[input >= 0.866025403784438646]
        grad_input[input <= -0.866025403784438646] = 2*input[input <= -0.866025403784438646]
        return grad_input*grad_output

if __name__ == '__main__':
    line = torch.nn.Sequential(
        torch.nn.Linear(16,32),
        torch.nn.ReLU(),
        torch.nn.Linear(32,64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2)
    )
    x = torch.randn(512,16)
    y = torch.randn()
