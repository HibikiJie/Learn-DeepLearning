from torchvision.models import vgg19, vgg16
from torch import nn
from torchvision.utils import save_image
import torch
import cv2


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        a = vgg19(True)
        a = a.features
        self.layer1 = a[:4]
        self.layer2 = a[4:9]
        self.layer3 = a[9:18]
        self.layer4 = a[18:27]
        self.layer5 = a[27:36]

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out1, out2, out3, out4, out5


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        a = vgg16(True)
        a = a.features
        self.layer1 = a[:4]
        self.layer2 = a[4:9]
        self.layer3 = a[9:16]
        self.layer4 = a[16:23]
        self.layer5 = a[23:30]

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out1, out2, out3, out4, out5


class GNet(nn.Module):
    def __init__(self, image, a=1e-20):
        super(GNet, self).__init__()
        # image = torch.log(image / ((1 - image) + a))
        self.image_g = nn.Parameter(image.detach().clone())

    def forward(self):
        # return torch.sigmoid(self.image_g)
        return self.image_g.clamp(0, 1)


def load_image(path):
    image = cv2.imread(path)
    # image = cv2.resize(image, (256,256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float() / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image


def get_gram_matrix(f_map):
    """
    获取格拉姆矩阵
    :param f_map:特征图
    :return:格拉姆矩阵，形状（通道数,通道数）
    """
    n, c, h, w = f_map.shape
    if n == 1:
        f_map = f_map.reshape(c, h * w)
        gram_matrix = torch.mm(f_map, f_map.t())
        return gram_matrix
    else:
        raise ValueError('批次应该为1,但是传入的不为1')


# def transfer():
#
#     import torch
#     import torch.nn as nn
#     import torch.nn.functional as F
#     import torch.optim as optim
#
#     from PIL import Image
#     import matplotlib.pyplot as plt
#
#     import torchvision.transforms as transforms
#     import torchvision.models as models
#
#     import copy
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 所需的输出图像大小
#     imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
#
#     loader = transforms.Compose([
#         transforms.Resize(imsize),  # scale imported image
#         transforms.ToTensor()])  # transform it into a torch tensor
#
#     def image_loader(image_name):
#         image = Image.open(image_name)
#         # fake batch dimension required to fit network's input dimensions
#         image = loader(image).unsqueeze(0)
#         return image.to(device, torch.float)
#
#     style_img = image_loader("s1.jpg")
#     content_img = image_loader("c1.jpg")
#     assert style_img.size() == content_img.size(), \
#         "we need to import style and content images of the same size"
#     # 现在，让我们创建一个方法，通过重新将图片转换成PIL格式来展示，并使用plt.imshow展示它的拷贝。我们将尝试展示内容和风格图片来确保它们被正确的导入。
#     unloader = transforms.ToPILImage()  # reconvert into PIL image
#     plt.ion()
#
#     def imshow(tensor, title=None):
#         image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
#         image = image.squeeze(0)  # remove the fake batch dimension
#         image = unloader(image)
#         plt.imshow(image)
#         if title is not None:
#             plt.title(title)
#         plt.pause(0.001)  # pause a bit so that plots are updated
#
#     plt.figure()
#     imshow(style_img, title='Style Image')
#     plt.figure()
#     imshow(content_img, title='Content Image')
#
#     class ContentLoss(nn.Module):
#         def __init__(self, target, ):
#             super(ContentLoss, self).__init__()
#             # 我们从用于动态计算梯度的树中“分离”目标内容：
#             # 这是一个声明的值，而不是变量。
#             # 否则标准的正向方法将引发错误。
#             self.target = target.detach()
#
#         def forward(self, input):
#             self.loss = F.mse_loss(input, self.target)
#             return input
#
#     def gram_matrix(input):
#         a, b, c, d = input.size()  # a=batch size(=1)
#         # 特征映射 b=number
#         # (c,d)=dimensions of a f. map (N=c*d)
#
#         features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
#
#         G = torch.mm(features, features.t())  # compute the gram product
#
#         # 我们通过除以每个特征映射中的元素数来“标准化”gram矩阵的值.
#         return G.div(a * b * c * d)
#
#     class StyleLoss(nn.Module):
#
#         def __init__(self, target_feature):
#             super(StyleLoss, self).__init__()
#             self.target = gram_matrix(target_feature).detach()
#
#         def forward(self, input):
#             G = gram_matrix(input)
#             self.loss = F.mse_loss(G, self.target)
#             return input
#
#     cnn = models.vgg19(pretrained=True).features.to(device).eval()
#
#     cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
#     cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
#
#     # 创建一个模块来规范化输入图像
#     # 这样我们就可以轻松地将它放入nn.Sequential中
#     class Normalization(nn.Module):
#         def __init__(self, mean, std):
#             super(Normalization, self).__init__()
#             # .view the mean and std to make them [C x 1 x 1] so that they can
#             # directly work with image Tensor of shape [B x C x H x W].
#             # B is batch size. C is number of channels. H is height and W is width.
#             self.mean = torch.tensor(mean).view(-1, 1, 1).clone().clone().detach()
#             self.std = torch.tensor(std).view(-1, 1, 1).clone().detach()
#
#         def forward(self, img):
#             # normalize img
#             return (img - self.mean) / self.std
#
#     content_layers_default = ['conv_4']
#     style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
#
#     def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
#                                    style_img, content_img,
#                                    content_layers=content_layers_default,
#                                    style_layers=style_layers_default):
#         cnn = copy.deepcopy(cnn)
#
#         # 规范化模块
#         normalization = Normalization(normalization_mean, normalization_std).to(device)
#
#         # 只是为了拥有可迭代的访问权限或列出内容/系统损失
#         content_losses = []
#         style_losses = []
#
#         # 假设cnn是一个`nn.Sequential`，
#         # 所以我们创建一个新的`nn.Sequential`来放入应该按顺序激活的模块
#         model = nn.Sequential(normalization)
#
#         i = 0  # increment every time we see a conv
#         for layer in cnn.children():
#             if isinstance(layer, nn.Conv2d):
#                 i += 1
#                 name = 'conv_{}'.format(i)
#             elif isinstance(layer, nn.ReLU):
#                 name = 'relu_{}'.format(i)
#                 # 对于我们在下面插入的`ContentLoss`和`StyleLoss`，
#                 # 本地版本不能很好地发挥作用。所以我们在这里替换不合适的
#                 layer = nn.ReLU(inplace=False)
#             elif isinstance(layer, nn.MaxPool2d):
#                 name = 'pool_{}'.format(i)
#             elif isinstance(layer, nn.BatchNorm2d):
#                 name = 'bn_{}'.format(i)
#             else:
#                 raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
#
#             model.add_module(name, layer)
#
#             if name in content_layers:
#                 # 加入内容损失:
#                 target = model(content_img).detach()
#                 content_loss = ContentLoss(target)
#                 model.add_module("content_loss_{}".format(i), content_loss)
#                 content_losses.append(content_loss)
#
#             if name in style_layers:
#                 # 加入风格损失:
#                 target_feature = model(style_img).detach()
#                 style_loss = StyleLoss(target_feature)
#                 model.add_module("style_loss_{}".format(i), style_loss)
#                 style_losses.append(style_loss)
#
#         # 现在我们在最后的内容和风格损失之后剪掉了图层
#         for i in range(len(model) - 1, -1, -1):
#             if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
#                 break
#
#         model = model[:(i + 1)]
#
#         return model, style_losses, content_losses
#
#     input_img = content_img.clone()
#     # 如果您想使用白噪声而取消注释以下行：
#     # input_img = torch.randn(content_img.data.size(), device=device)
#
#     # 将原始输入图像添加到图中：
#     plt.figure()
#     imshow(input_img, title='Input Image')
#
#     def get_input_optimizer(input_img):
#         # 此行显示输入是需要渐变的参数
#         optimizer = optim.LBFGS([input_img.requires_grad_()])
#         return optimizer
#
#     def run_style_transfer(cnn, normalization_mean, normalization_std,
#                            content_img, style_img, input_img, num_steps=300,
#                            style_weight=1000000, content_weight=1):
#         """Run the style transfer."""
#         print('Building the style transfer model..')
#         model, style_losses, content_losses = get_style_model_and_losses(cnn,
#                                                                          normalization_mean, normalization_std,
#                                                                          style_img,
#                                                                          content_img)
#         optimizer = get_input_optimizer(input_img)
#
#         print('Optimizing..')
#         run = [0]
#         while run[0] <= num_steps:
#
#             def closure():
#                 # 更正更新的输入图像的值
#                 input_img.data.clamp_(0, 1)
#
#                 optimizer.zero_grad()
#                 model(input_img)
#                 style_score = 0
#                 content_score = 0
#
#                 for sl in style_losses:
#                     style_score += sl.loss
#                 for cl in content_losses:
#                     content_score += cl.loss
#
#                 style_score *= style_weight
#                 content_score *= content_weight
#
#                 loss = style_score + content_score
#                 loss.backward()
#
#                 run[0] += 1
#                 if run[0] % 50 == 0:
#                     print("run {}:".format(run))
#                     print('Style Loss : {:4f} Content Loss: {:4f}'.format(
#                         style_score.item(), content_score.item()))
#                     print()
#
#                 return style_score + content_score
#
#             optimizer.step(closure)
#
#         # 最后的修正......
#         input_img.data.clamp_(0, 1)
#
#         return input_img
#
#     output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
#                                 content_img, style_img, input_img)
#
#     plt.figure()
#     imshow(output, title='Output Image')
#
#     # sphinx_gallery_thumbnail_number = 4
#     plt.ioff()
#     plt.show()


if __name__ == '__main__':
    image_content = load_image('c1.jpg').cuda()
    image_style = load_image('4.jpg').cuda()
    net = VGG19().cuda()
    g_net = GNet(image_content).cuda()
    optimizer = torch.optim.Adam(g_net.parameters())
    loss_func = nn.MSELoss().cuda()

    """计算分格,并计算gram矩阵"""
    s1, s2, s3, s4, s5 = net(image_style)
    s1 = get_gram_matrix(s1).detach().clone()
    s2 = get_gram_matrix(s2).detach().clone()
    s3 = get_gram_matrix(s3).detach().clone()
    s4 = get_gram_matrix(s4).detach().clone()
    s5 = get_gram_matrix(s5).detach().clone()
    """计算内容"""
    c1, c2, c3, c4, c5 = net(image_content)
    c1 = c1.detach().clone()
    c2 = c2.detach().clone()
    c3 = c3.detach().clone()
    c4 = c4.detach().clone()
    c5 = c5.detach().clone()
    i = 0
    while True:
        """生成图片，计算损失"""
        image = g_net()
        out1, out2, out3, out4, out5 = net(image)

        """计算分格损失"""
        loss_s1 = loss_func(get_gram_matrix(out1), s1)
        loss_s2 = loss_func(get_gram_matrix(out2), s2)
        loss_s3 = loss_func(get_gram_matrix(out3), s3)
        loss_s4 = loss_func(get_gram_matrix(out4), s4)
        loss_s5 = loss_func(get_gram_matrix(out5), s5)
        loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4 + loss_s5

        """计算内容损失"""
        loss_c1 = loss_func(out1, c1)
        loss_c2 = loss_func(out2, c2)
        loss_c3 = loss_func(out3, c3)
        loss_c4 = loss_func(out4, c4)
        loss_c5 = loss_func(out5, c5)
        loss_c = 1 * loss_c1 + 1 * loss_c2 + 1 * loss_c3 + 1 * loss_c4 + 1 * loss_c5

        """总损失"""
        loss = 0*loss_c + loss_s

        """清空梯度、计算梯度、更新参数"""
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(i, loss.item(), loss_c.item(), loss_s.item())
        if i % 1000 == 0:
            save_image(image, f'{i}.jpg', padding=0, normalize=True, range=(0, 1))
        i += 1
