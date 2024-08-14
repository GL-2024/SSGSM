import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fusion_strategy


class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out


# light version
class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []

        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

def fusion_calculate( I0, DOLP,img_Imask_sp,img_Imask_glant):
    scaled_mask_sp = F.interpolate(img_Imask_sp, size=I0.shape[2:], mode='bilinear', align_corners=False)
    scaled_mask_glant = F.interpolate(img_Imask_glant, size=I0.shape[2:], mode='bilinear', align_corners=False)
    scaled_weight_I0=0.2*(1-scaled_mask_glant)+0.5*(scaled_mask_glant-scaled_mask_sp)
    scaled_weight_DOLP =0.8*(1-scaled_mask_glant)+0.5*(scaled_mask_glant-scaled_mask_sp)+scaled_mask_sp
    # 对特征图应用权重
    weighted_I0 = I0 * scaled_weight_I0
    weighted_DOLP = DOLP* scaled_weight_DOLP

    # 将加权后的特征图进行融合
    # fused_feature = torch.cat([weighted_I0, weighted_DOLP], dim=1)  # 在通道维度上连接两个加权特征图
    fused_feature = weighted_I0 + weighted_DOLP
    return fused_feature
"""
def CMDAF( I0, AOP, img_Imask_glant):  # I0[0], AOLP[0], img_Imask_glant
    img_Imask_glant = F.interpolate(img_Imask_glant, size=I0.shape[2:], mode='bilinear', align_corners=False)
    sub_I0_AOP = I0 - AOP
    sub_I0_AOP = sub_I0_AOP * (1 - img_Imask_glant)
    sub_w_I0_AOP = torch.mean(sub_I0_AOP, dim=[1, 2], keepdim=True)  # Global Average Pooling
    w_I0_AOP = torch.sigmoid(sub_w_I0_AOP)

    sub_AOP_I0 = AOP - I0
    sub_AOP_I0 = sub_AOP_I0 * (1 - img_Imask_glant)
    sub_w_AOP_I0 = torch.mean(sub_AOP_I0, dim=[1, 2], keepdim=True)  # Global Average Pooling
    w_AOP_I0 = torch.sigmoid(sub_w_AOP_I0)

    F_dI0 = w_I0_AOP * sub_AOP_I0  # 放大差分信号，此处是否应该调整为sub_ir_vi
    F_dAOP = w_AOP_I0 * sub_I0_AOP

    F_fI0 = I0 * (1 - img_Imask_glant) + F_dAOP
    F_fAOP = AOP + F_dI0
    F_concatenated = torch.cat((F_fAOP, F_fI0), dim=1)
    return F_concatenated
"""

class CMDAF_layer(nn.Module):#def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True):
    def __init__(self):
        super(CMDAF_layer, self).__init__()
    """
    def CMDAF_2(self,  AOP,DOLP, img_Imask_glant):
        img_Imask_glant = F.interpolate(img_Imask_glant, size=AOP.shape[2:], mode='bilinear', align_corners=False)

        sub_DOLP_AOP = DOLP - AOP
        sub_DOLP_AOP=sub_DOLP_AOP*(1-img_Imask_glant)
        sub_w_DOLP_AOP = torch.mean(sub_DOLP_AOP, dim=[1, 2], keepdim=True)  # Global Average Pooling
        w_DOLP_AOP = torch.sigmoid(sub_w_DOLP_AOP)

        sub_AOP_DOLP = AOP - DOLP
        sub_AOP_DOLP = sub_AOP_DOLP * (1 - img_Imask_glant)
        sub_w_AOP_DOLP = torch.mean(sub_AOP_DOLP, dim=[1, 2], keepdim=True)  # Global Average Pooling
        w_AOP_DOLP = torch.sigmoid(sub_w_AOP_DOLP)


        #F_dI0 = w_I0_AOP  * sub_I0_AOP  # 放大差分信号，此处是否应该调整为sub_ir_vi
        #F_dAOP = w_AOP_I0 * sub_AOP_I0

        F_fAOP = AOP +w_DOLP_AOP*sub_DOLP_AOP
        F_fDOLP=DOLP*(1-img_Imask_glant)+w_AOP_DOLP*sub_AOP_DOLP

        return F_fAOP,F_fDOLP
    """

    def CMDAF(self, I0, AOP,DOLP, img_Imask_glant):
        img_Imask_glant = F.interpolate(img_Imask_glant, size=I0.shape[2:], mode='bilinear', align_corners=False)
        sub_I0_AOP = I0 - AOP
        sub_w_I0_AOP = torch.mean(sub_I0_AOP, dim=[1, 2], keepdim=True)  # Global Average Pooling
        w_I0_AOP = torch.sigmoid(sub_w_I0_AOP)

        sub_AOP_I0 = AOP - I0
        sub_w_AOP_I0 = torch.mean(sub_AOP_I0, dim=[1, 2], keepdim=True)  # Global Average Pooling
        w_AOP_I0 = torch.sigmoid(sub_w_AOP_I0)

        sub_DOLP_AOP = DOLP - AOP
        sub_DOLP_AOP=sub_DOLP_AOP*(1-img_Imask_glant)
        sub_w_DOLP_AOP = torch.mean(sub_DOLP_AOP, dim=[1, 2], keepdim=True)  # Global Average Pooling
        w_DOLP_AOP = torch.sigmoid(sub_w_DOLP_AOP)

        sub_AOP_DOLP = AOP - DOLP
        sub_AOP_DOLP = sub_AOP_DOLP * (1 - img_Imask_glant)
        sub_w_AOP_DOLP = torch.mean(sub_AOP_DOLP, dim=[1, 2], keepdim=True)  # Global Average Pooling
        w_AOP_DOLP = torch.sigmoid(sub_w_AOP_DOLP)

        sub_I0_DOLP = I0 - DOLP
        sub_I0_DOLP = sub_I0_DOLP * (1 - img_Imask_glant)
        sub_w_I0_DOLP = torch.mean(sub_I0_DOLP, dim=[1, 2], keepdim=True)  # Global Average Pooling
        w_I0_DOLP = torch.sigmoid(sub_w_I0_DOLP)

        sub_DOLP_I0 = DOLP - I0
        sub_DOLP_I0 = sub_DOLP_I0 * (1 - img_Imask_glant)
        sub_w_DOLP_I0 = torch.mean(sub_DOLP_I0, dim=[1, 2], keepdim=True)  # Global Average Pooling
        w_DOLP_I0 = torch.sigmoid(sub_w_DOLP_I0)

        #F_dI0 = w_I0_AOP  * sub_I0_AOP  # 放大差分信号，此处是否应该调整为sub_ir_vi
        #F_dAOP = w_AOP_I0 * sub_AOP_I0

        F_fI0 = I0  + w_AOP_I0 * sub_AOP_I0+w_DOLP_I0*sub_DOLP_I0
        F_fAOP = AOP + w_I0_AOP*sub_I0_AOP+w_DOLP_AOP*sub_DOLP_AOP
        F_fDOLP=DOLP*(1-img_Imask_glant)+w_I0_DOLP*sub_I0_DOLP+w_AOP_DOLP*sub_AOP_DOLP

        return F_fAOP, F_fI0,F_fDOLP

class GradientModel(nn.Module):
    def __init__(self):
        super(GradientModel, self).__init__()
        self.layer_laplacian = LaplacianLayer()
        self.layer_sobel = SobelLayer()
        self.layer_1_conv1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.layer_1_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.layer_2_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.layer_3_conv = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.batch_norm4 = nn.BatchNorm2d(128)

    def forward(self, feature_fusion):
        feature_fusion_laplacian = self.layer_laplacian(feature_fusion)
        feature_new1 = feature_fusion + feature_fusion_laplacian
        feature_fusion_sobel = self.layer_sobel(feature_fusion)
        feature_fusion_sobel_new = self.batch_norm1(self.layer_1_conv1(feature_fusion_sobel))
        conv1 = self.batch_norm2(self.layer_1_conv2(feature_new1))
        conv1 = nn.functional.leaky_relu(conv1)
        conv2 = self.batch_norm3(self.layer_2_conv(conv1))
        conv2 = nn.functional.leaky_relu(conv2)
        conv3 = self.batch_norm4(self.layer_3_conv(conv2))
        feature_fusion_gradient = torch.cat([conv3, feature_fusion_sobel_new], dim=1)
        return feature_fusion_gradient


class LaplacianLayer(nn.Module):
    def __init__(self):
        super(LaplacianLayer, self).__init__()
        # Define the laplacian kernel
        self.kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32)
        self.kernel = self.kernel.view(1, 1, 3, 3)

    def forward(self, x):
        # Apply laplacian filter
        gradient_orig = torch.abs(F.conv2d(x, self.kernel, stride=1, padding=1))
        grad_min = torch.min(gradient_orig)
        grad_max = torch.max(gradient_orig)
        grad_norm = (gradient_orig - grad_min) / (grad_max - grad_min + 0.0001)
        return grad_norm


class SobelLayer(nn.Module):
    def __init__(self):
        super(SobelLayer, self).__init__()
        # Define the sobel kernels
        smooth_kernel_x = torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32)
        smooth_kernel_y = smooth_kernel_x.transpose(0, 1)
        self.kernel_x = smooth_kernel_x.view(1, 1, 2, 2)
        self.kernel_y = smooth_kernel_y.view(1, 1, 2, 2)

    def forward(self, x):
        # Apply sobel filter
        gradient_orig_x = torch.abs(F.conv2d(x, self.kernel_x, stride=1, padding=1))
        gradient_orig_y = torch.abs(F.conv2d(x, self.kernel_y, stride=1, padding=1))
        grad_min_x = torch.min(gradient_orig_x)
        grad_max_x = torch.max(gradient_orig_x)
        grad_min_y = torch.min(gradient_orig_y)
        grad_max_y = torch.max(gradient_orig_y)
        grad_norm_x = (gradient_orig_x - grad_min_x) / (grad_max_x - grad_min_x + 0.0001)
        grad_norm_y = (gradient_orig_y - grad_min_y) / (grad_max_y - grad_min_y + 0.0001)
        grad_norm = grad_norm_x + grad_norm_y
        return grad_norm

# NestFuse network - light, no desnse
class NestFuse_autoencoder(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True):
        super(NestFuse_autoencoder, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1
        self.conv = nn.ModuleList([
            nn.Conv2d(3 * nb_filter[0], nb_filter[0], kernel_size=1),
            nn.Conv2d(3 * nb_filter[1], nb_filter[1], kernel_size=1),
            nn.Conv2d(3 * nb_filter[2], nb_filter[2], kernel_size=1),
            nn.Conv2d(3 * nb_filter[3], nb_filter[3], kernel_size=1)
        ])

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        # encoder
        self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)
        self.DB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.DB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.DB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.DB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)
        # self.DB5_0 = block(nb_filter[3], nb_filter[4], kernel_size, 1)

        # decoder
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)

        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)

        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)

        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)

    def encoder(self, input):
        x = self.conv0(input)
        x1_0 = self.DB1_0(x)
        x2_0 = self.DB2_0(self.pool(x1_0))
        x3_0 = self.DB3_0(self.pool(x2_0))
        x4_0 = self.DB4_0(self.pool(x3_0))
            # x5_0 = self.DB5_0(self.pool(x4_0))
        return [x1_0, x2_0, x3_0, x4_0]

    def fusion(self, en1, en2, p_type):
        # attention weight
        fusion_function = fusion_strategy.attention_fusion_weight

        f1_0 = fusion_function(en1[0], en2[0], p_type)
        f2_0 = fusion_function(en1[1], en2[1], p_type)
        f3_0 = fusion_function(en1[2], en2[2], p_type)
        f4_0 = fusion_function(en1[3], en2[3], p_type)
        return [f1_0, f2_0, f3_0, f4_0]
    """
    def fusion_calculate(self,I0,DOLP,weight,i):
        scaled_weight = F.interpolate(weight, size=I0[i].shape[2:], mode='bilinear', align_corners=False)

        # 对特征图应用权重
        weighted_I0 = I0[i] * scaled_weight
        weighted_DOLP = DOLP[i] * scaled_weight

        # 将加权后的特征图进行融合
        # fused_feature = torch.cat([weighted_I0, weighted_DOLP], dim=1)  # 在通道维度上连接两个加权特征图
        fused_feature = weighted_I0 + weighted_DOLP
        return fused_feature
    """

    def fusion_strage(self,I0,AOLP,DOLP,img_Imask_glant):
        cmdaf_layer = CMDAF_layer()
        F_fAOP, F_fI0,F_fDOLP=cmdaf_layer.CMDAF(I0[0],AOLP[0],DOLP[0],img_Imask_glant)
        #F_concatenated = torch.cat((F_fAOP, F_fI0,F_fDOLP), dim=1)
        # 使用卷积层来减少通道数
        #fuse_1 = self.conv[0](F_concatenated)
        fuse_1=F_fAOP+F_fI0+F_fDOLP

        F_fAOP, F_fI0,F_fDOLP = cmdaf_layer.CMDAF(I0[1],AOLP[1],DOLP[1],img_Imask_glant)
        #F_concatenated = torch.cat((F_fAOP, F_fI0,F_fDOLP), dim=1)
        #fuse_2 = self.conv[1](F_concatenated)
        fuse_2=F_fAOP+F_fI0+F_fDOLP

        F_fAOP, F_fI0,F_fDOLP = cmdaf_layer.CMDAF(I0[2],AOLP[2],DOLP[2],img_Imask_glant)
        #F_concatenated = torch.cat((F_fAOP, F_fI0,F_fDOLP), dim=1)
        #fuse_3 = self.conv[2](F_concatenated)
        fuse_3 = F_fAOP + F_fI0 + F_fDOLP
        F_fAOP, F_fI0,F_fDOLP = cmdaf_layer.CMDAF(I0[3],AOLP[3],DOLP[3],img_Imask_glant)
        #F_concatenated = torch.cat((F_fAOP, F_fI0,F_fDOLP), dim=1)
        #fuse_4 = self.conv[3](F_concatenated)
        fuse_4 = F_fAOP + F_fI0 + F_fDOLP
        #fuse_1 = fusion_calculate(I0[0], DOLP[0], img_Imask_sp,img_Imask_glant)
        #fuse_2 = fusion_calculate(I0[1], DOLP[1], img_Imask_sp,img_Imask_glant )
        #fuse_3 = fusion_calculate(I0[2], DOLP[2], img_Imask_sp,img_Imask_glant)
        #fuse_4 = fusion_calculate(I0[3], DOLP[3], img_Imask_sp,img_Imask_glant)
        return [fuse_1, fuse_2, fuse_3, fuse_4]



    def decoder_train(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)#由64通道变为1通道
            return [output]

    def decoder_eval(self, f_en):

        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]