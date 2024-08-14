import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import torch.nn as nn
import cv2
import net_tre_class as networks
from args_fusion import args

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class Textureloss(nn.Module):#DOLP_input,AOP_input,Imask_glant_input
    def __init__(self):
        super(Textureloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_fuse,image_I0,image_AOP,image_DOLP):
        I0_grad = self.sobelconv(image_I0)
        DOLP_grad = self.sobelconv(image_DOLP)
        fuse_img_grad = self.sobelconv(image_fuse)
        AOP_grad = self.sobelconv(image_AOP)
        grad_joint1 = torch.min(DOLP_grad, AOP_grad)
        #grad_joint2 = torch.min(I0_grad, AOP_grad)
        ##非区域耀光
        #x_grad_joint = torch.min(I0_grad*(1-image_Imask_glant), torch.min(DOLP_grad*(1-image_Imask_glant),AOP_grad*(1-image_Imask_glant)))
        #x_grad_joint = torch.min(I0_grad ,torch.min(DOLP_grad , AOP_grad ))
        x_grad_joint = torch.max(grad_joint1, I0_grad)
        ##耀光区域 image_Imask_glant
        #y_grad_joint = torch.min(I0_grad*image_Imask_glant,AOP_grad*image_Imask_glant)
        #y_grad_joint = torch.min(I0_grad , AOP_grad )
        #loss_tex_x = F.l1_loss( fuse_img_grad*(1-image_Imask_glant),x_grad_joint)
        loss_tex_x = F.l1_loss(fuse_img_grad , x_grad_joint)
        #loss_tex_y = F.l1_loss( fuse_img_grad * image_Imask_glant, y_grad_joint)
        #loss_tex_y = F.l1_loss(fuse_img_grad , y_grad_joint)
        #loss_tex=loss_tex_x+loss_tex_y
        loss_tex= torch.mean(loss_tex_x)
        return loss_tex

class Intenloss(nn.Module):
    def __init__(self):
        super(Intenloss, self).__init__()
        self.sobelconv = Sobelxy()
    def forward(self, output, outputs_Iin,AOP_input,DOLP):
        # 定义均值滤波器
        def mean_filter(input_tensor, kernel_size=3):
            pad_size = kernel_size // 2
            # 使用padding来处理边缘情况
            padded_input = nn.functional.pad(input_tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
            # 定义一个均值滤波的卷积核
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=input_tensor.device) / (kernel_size ** 2)
            # 对输入进行卷积操作，即均值滤波
            filtered_output = nn.functional.conv2d(padded_input, kernel)
            return filtered_output

        # 对 output 和 DOLP_input 进行均值滤波
        #filtered_output = mean_filter(output)
        #filtered_AOP_input = mean_filter(AOP_input)
        AOP_grad = self.sobelconv(AOP_input)
        DOLP_grad = self.sobelconv(DOLP)
        AOP_grad_flit=mean_filter(AOP_grad)
        DOLP_grad_flit = mean_filter(DOLP_grad)
        mask = torch.where(AOP_grad_flit > DOLP_grad_flit, torch.tensor(0, device=AOP_grad_flit.device),
                           torch.tensor(1, device=AOP_grad_flit.device))

        inten_min=mask*DOLP+(1-mask)*outputs_Iin
        """
        inten_min = torch.zeros_like(AOP_input)

        for i in range(0, AOP_grad.shape[2], 2):  # 假设AOP_grad.shape[2]能被3整除
            for j in range(0, AOP_grad.shape[3], 2):  # 假设AOP_grad.shape[3]能被3整除
                AOP_block = AOP_grad[:, :, i:i + 2, j:j + 2]
                DOLP_block = DOLP_grad[:, :, i:i + 2, j:j + 2]

                mean_AOP = torch.mean(AOP_block).item()
                mean_DOLP = torch.mean(DOLP_block).item()

                if mean_AOP > mean_DOLP:
                    inten_min[:, :, i:i + 2, j:j + 2] = torch.min(outputs_Iin[:, :, i:i + 2, j:j + 2],
                                                                  DOLP[:, :, i:i + 2, j:j + 2])
                elif mean_AOP < mean_DOLP:
                    inten_min[:, :, i:i + 2, j:j + 2] = torch.max(outputs_Iin[:, :, i:i + 2, j:j + 2],
                                                                  DOLP[:, :, i:i + 2, j:j + 2])
        """

        """
        AOP2_show = filtered_AOP_input.squeeze(1).squeeze(1).cpu().numpy()
        first_image = AOP2_show[0]
        normalized_image = (first_image - np.min(first_image)) / (np.max(first_image) - np.min(first_image))
        # 将数据转换为 uint8 类型
        binary_image_np = (normalized_image * 255).astype(np.uint8)

        # 显示图像
        cv2.imshow('Binary Image', binary_image_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        # 计算滤波后的张量之间的 L1 损失
        #loss = nn.L1Loss()
        #inten_min = torch.max(outputs_Iin,AOP_input)

        inten_loss_value=F.l1_loss(output, inten_min)
        inten_loss_value = torch.mean(inten_loss_value)
        return inten_loss_value
    """

    def forward(self, image_fuse,image_I0,image_DOLP,image_AOP,image_Imask_glant):
        nor_glant = torch.max(image_I0 * (1 - image_Imask_glant),
                                 torch.max(image_DOLP * (1 - image_Imask_glant), image_AOP * (1 - image_Imask_glant)))
        glant = torch.max(image_I0 * image_Imask_glant, image_AOP * image_Imask_glant)
        loss_nor_glant = F.l1_loss(image_fuse, nor_glant)
        loss_glant = F.l1_loss(image_fuse, glant)
        loss = loss_nor_glant + loss_glant
        loss = torch.mean(loss)
        return loss
    """

class vgg_loss(nn.Module):
    def __init__(self):
        super(vgg_loss, self).__init__()
        self.gpu_ids = [0]
        self.vgg_loss = networks.PerceptualLoss()
        self.vgg_loss.cuda()
        self.vgg = networks.load_vgg16("./model", self.gpu_ids)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self,fake_B,real_A):
        loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg,fake_B,real_A)
        return loss_vgg_b

class Maskpixelloss(nn.Module):
    def __init__(self):
        super(Maskpixelloss, self).__init__()

    def forward(self,image_fuse,image_I0,image_DOLP,Imask_sp_input,Imask_glant_input):
        a1=0.5
        a2=0.5
        b1=0.8
        b2=0.2
        loss_saturation=F.l1_loss(Imask_sp_input*image_fuse, Imask_sp_input*image_I0)
        loss_glare_I0=F.l1_loss((Imask_glant_input-Imask_sp_input)*image_fuse,(Imask_glant_input-Imask_sp_input)*image_I0)
        loss_glare_DOLP = F.l1_loss((Imask_glant_input-Imask_sp_input) * image_fuse, (Imask_glant_input-Imask_sp_input) * image_DOLP)
        loss_no_glare_I0=F.l1_loss((1-Imask_glant_input)*image_fuse,(1-Imask_glant_input)*image_I0)
        loss_no_glare_DOLP = F.l1_loss((1 - Imask_glant_input) * image_fuse, (1 - Imask_glant_input) * image_DOLP)
        loss_Maskpixel=loss_saturation+a1*loss_glare_I0+a2*loss_glare_DOLP+b2*loss_no_glare_I0+b1*loss_no_glare_DOLP
        return loss_Maskpixel

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
