import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""创建模型"""
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""实例化模型"""
model = LeNet().to(device=device)

"""检查模块，它将包含两个参数weight和bias，并且没有缓冲区。"""
module = model.conv1
print(list(module.named_parameters()))
print(list(module.named_buffers()))

"""
剪枝模块model.conv1，中的weight，amount指定了剪枝的百分比。
如果是非负整数，则是剪去数量。
"""
prune.random_unstructured(module, name="weight", amount=0.3)

"""剪枝是通过生成一个mask掩码来进行的，这将被保存到named_buffers中"""
print(list(module.named_buffers()))

print(module.weight)
"""每剪一次，可获得一个钩子对象。"""
print(module._forward_pre_hooks)

"""迭代修剪"""
"""
一个模块中的同一参数可以被多次修剪，各种修剪调用的效果等于串联应用的各种蒙版的组合。"""
"""n为范数"""
prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
print(module.weight)

"""修剪的参数，永久化"""
prune.remove(module, 'weight')
print(list(module.named_parameters()))

"""修剪模型中的多个参数"""
new_model = LeNet()
for name, module in new_model.named_modules():
    # prune 20% of connections in all 2D-conv layers
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.2)
    # prune 40% of connections in all linear layers
    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.4)


"""全局剪枝"""
model = LeNet()

parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)



