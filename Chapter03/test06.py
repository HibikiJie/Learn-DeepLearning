import torch
from torch.autograd import Variable
class MyArccos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i.acos()

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        print(input_)
        grad_input = grad_output.clone()
        print(grad_input.data)
        print(-0.9999<input_<0.9999)
        grad_input[-0.9999<input_<0.9999] = -1/((1-grad_input**2)**0.5)
        grad_input[input_ < -0.9999] = -70.7048
        grad_input[input_ > 0.9999] = -70.7048
        print(grad_output,11)
        print(input_)
        print(grad_input,22)
        return grad_input

input_=torch.randn([1],dtype=torch.float32,requires_grad=True)
print(input_)
arccos = MyArccos.apply(input_)
# out = arccos(input_)
print(arccos)
arccos.backward()
print(input_.grad)