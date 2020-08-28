import torch


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss with respect to the output, and we need to compute the gradient of the loss with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


x = torch.randn(3,3,3,dtype=torch.float32,requires_grad=True)
relu = MyReLU.apply
out = relu(x)
print(out)
out.backward()
print(x.grad)

def forward(self, features, target):
    target = target.unsqueeze(dim=1)
    features = torch.nn.functional.normalize(features, dim=1)
    w = torch.nn.functional.normalize(self.w, dim=0)
    cos_theta = torch.matmul(features, w)
    cos_theta_plus = cos_theta-1.0
    top = torch.exp(cos_theta_plus).gather(dim=1, index=target)
    down_tempy = torch.exp(cos_theta)
    down = down_tempy.sum(dim=1, keepdim=True) - down_tempy.gather(dim=1, index=target) + top
    return -torch.log(top / down).sum() / len(target)