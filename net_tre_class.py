import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fusion_strategy
import os

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

class CMDAF_layer(nn.Module):#def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True):
    def __init__(self):
        super(CMDAF_layer, self).__init__()

        def CMDAF(self, I0, AOP, DOLP, img_Imask_glant):
            img_Imask_glant = F.interpolate(img_Imask_glant, size=I0.shape[2:], mode='bilinear', align_corners=False)
            sub_I0_AOP = I0 - AOP
            sub_w_I0_AOP = torch.mean(sub_I0_AOP, dim=[1, 2], keepdim=True)  # Global Average Pooling
            w_I0_AOP = torch.sigmoid(sub_w_I0_AOP)

            sub_AOP_I0 = AOP - I0
            sub_w_AOP_I0 = torch.mean(sub_AOP_I0, dim=[1, 2], keepdim=True)  # Global Average Pooling
            w_AOP_I0 = torch.sigmoid(sub_w_AOP_I0)

            sub_DOLP_AOP = DOLP - AOP
            sub_DOLP_AOP = sub_DOLP_AOP * (1 - img_Imask_glant)
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

            # F_dI0 = w_I0_AOP  * sub_I0_AOP  # 放大差分信号，此处是否应该调整为sub_ir_vi
            # F_dAOP = w_AOP_I0 * sub_AOP_I0

            F_fI0 = I0 + w_AOP_I0 * sub_AOP_I0 + w_DOLP_I0 * sub_DOLP_I0
            F_fAOP = AOP + w_I0_AOP * sub_I0_AOP + w_DOLP_AOP * sub_DOLP_AOP
            F_fDOLP = DOLP * (1 - img_Imask_glant) + w_I0_DOLP * sub_I0_DOLP + w_AOP_DOLP * sub_AOP_DOLP

            return F_fAOP, F_fI0, F_fDOLP

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

class Encoder(nn.Module):
    def __init__(self, nb_filter, input_nc=1, deepsupervision=True):
        super(Encoder, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1

        # encoder
        self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)
        self.DB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.DB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.DB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.DB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)

    def forward(self, input_data):
        x0 = self.conv0(input_data)
        x1_0 = self.DB1_0(x0)
        x2_0 = self.DB2_0(x1_0)
        x3_0 = self.DB3_0(x2_0)
        x4_0 = self.DB4_0(x3_0)
        return [x0, x1_0, x2_0, x3_0, x4_0]

class fusion_strage(nn.Module):
    def __init__(self,nb_filter):
        super(fusion_strage, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv2d(3 * nb_filter[0], nb_filter[0], kernel_size=1),
            nn.Conv2d(3 * nb_filter[1], nb_filter[1], kernel_size=1),
            nn.Conv2d(3 * nb_filter[2], nb_filter[2], kernel_size=1),
            nn.Conv2d(3 * nb_filter[3], nb_filter[3], kernel_size=1)
        ])

    def forward(self, I0, AOLP, DOLP, img_Imask_glant):
        cmdaf_layer = CMDAF_layer()
        F_fAOP_1, F_fI0_1, F_fDOLP_1 = cmdaf_layer.CMDAF(I0[0], AOLP[0], DOLP[0], img_Imask_glant)
        F_concatenated = torch.cat((F_fAOP_1, F_fI0_1, F_fDOLP_1), dim=1)
        fuse_1 = self.conv[0](F_concatenated)
        F_fAOP_2, F_fI0_2, F_fDOLP_2 = cmdaf_layer.CMDAF(I0[1], AOLP[1], DOLP[1], img_Imask_glant)
        F_concatenated = torch.cat((F_fAOP_2, F_fI0_2, F_fDOLP_2), dim=1)
        fuse_2 = self.conv[1](F_concatenated)
        F_fAOP_3, F_fI0_3, F_fDOLP_3 = cmdaf_layer.CMDAF(I0[2], AOLP[2], DOLP[2], img_Imask_glant)
        F_concatenated = torch.cat((F_fAOP_3, F_fI0_3, F_fDOLP_3), dim=1)
        fuse_3 = self.conv[2](F_concatenated)
        F_fAOP_4, F_fI0_4, F_fDOLP_4 = cmdaf_layer.CMDAF(I0[3], AOLP[3], DOLP[3], img_Imask_glant)
        F_concatenated = torch.cat((F_fAOP_4, F_fI0_4, F_fDOLP_4), dim=1)
        fuse_4 = self.conv[3](F_concatenated)
        return [fuse_1, fuse_2, fuse_3, fuse_4]




def fusion_strage_de( I0,AOLP,DOLP,img_Imask_glant):
    cmdaf_layer = CMDAF_layer()
    F_fAOP, F_fI0, F_fDOLP = cmdaf_layer.CMDAF(I0[0], AOLP[0], DOLP[0], img_Imask_glant)
    fuse_1 = F_fAOP + F_fI0 + F_fDOLP
    F_fAOP, F_fI0, F_fDOLP = cmdaf_layer.CMDAF(I0[1], AOLP[1], DOLP[1], img_Imask_glant)
    fuse_2 = F_fAOP + F_fI0 + F_fDOLP
    F_fAOP, F_fI0, F_fDOLP = cmdaf_layer.CMDAF(I0[2], AOLP[2], DOLP[2], img_Imask_glant)
    fuse_3 = F_fAOP + F_fI0 + F_fDOLP
    F_fAOP, F_fI0, F_fDOLP = cmdaf_layer.CMDAF(I0[3], AOLP[3], DOLP[3], img_Imask_glant)
    fuse_4 = F_fAOP + F_fI0 + F_fDOLP
    return [fuse_1, fuse_2, fuse_3, fuse_4]


class CAIM(torch.nn.Module):
    def __init__(self, channels):
        super(CAIM, self).__init__()
        self.ca_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
        )

        self.ca_max = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),

        )
        self.sigmod = nn.Sigmoid()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.conv1 = nn.Conv2d(2 * channels, channels, 1, 1)

    def forward(self, x):
        # CA
        w_avg = self.ca_avg(x)
        w_max = self.ca_max(x)
        w = torch.cat([w_avg, w_max], dim=1)
        w = self.conv1(w)
        # SA
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x_sa = torch.cat([avgout, maxout], dim=1)
        x1 = self.conv(x_sa)
        output = self.sigmod(w * x1)
        return output
class Decoder(nn.Module):
    def __init__(self, nb_filter, output_nc=1, deepsupervision=True):
        super(Decoder, self).__init__()
        block = DenseBlock_light
        attention=CAIM
        kernel_size = 3

        self.up = nn.Upsample(scale_factor=2)
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)
        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)

        self.deepsupervision = deepsupervision
        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, 1)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, 1)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, 1)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, 1)

        self.conv = nn.ModuleList([
            nn.Conv2d(2 * nb_filter[0], nb_filter[0], kernel_size=1),
            nn.Conv2d(2 * nb_filter[1], nb_filter[1], kernel_size=1),
            nn.Conv2d(2 * nb_filter[2], nb_filter[2], kernel_size=1),
            nn.Conv2d(2 * nb_filter[3], nb_filter[3], kernel_size=1)
        ])

        self.atten_1=attention(nb_filter[0])
        self.atten_2 = attention(nb_filter[1])
        self.atten_3 = attention(nb_filter[2])
        self.atten_4 = attention(nb_filter[3])


    def forward(self, I0, AOLP):
        #features=self.fuse(I0, AOLP, DOLP, img_Imask_glant)
        features = self.fuse(I0, AOLP)
        x1_1 = self.DB1_1(torch.cat([features[0], self.up(features[1])], 1))
        x2_1 = self.DB2_1(torch.cat([features[1], self.up(features[2])], 1))
        x1_2 = self.DB1_2(torch.cat([features[0], x1_1, self.up(x2_1)], 1))
        x3_1 = self.DB3_1(torch.cat([features[2], self.up(features[3])], 1))
        x2_2 = self.DB2_2(torch.cat([features[1], x2_1, self.up(x3_1)], 1))
        x1_3 = self.DB1_3(torch.cat([features[0], x1_1, x1_2, self.up(x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            #return [output ,features[4] ,features[5]]
            #return [output, I0[2], AOLP[2]]
            return [output]

    def fuse(self, I0, AOLP):

        F_fAOP_1, F_fI0_1 = self.CMDAF_1(I0[0], AOLP[0])
        F_concatenated = torch.cat((F_fAOP_1, F_fI0_1), dim=1)
        fuse_1 = self.conv[0](F_concatenated)
        F_fAOP_2, F_fI0_2 = self.CMDAF_2(I0[1], AOLP[1])
        F_concatenated = torch.cat((F_fAOP_2, F_fI0_2), dim=1)
        fuse_2 = self.conv[1](F_concatenated)
        F_fAOP_3, F_fI0_3= self.CMDAF_3(I0[2], AOLP[2])
        F_concatenated = torch.cat((F_fAOP_3, F_fI0_3), dim=1)
        fuse_3 = self.conv[2](F_concatenated)
        F_fAOP_4, F_fI0_4 = self.CMDAF_4(I0[3], AOLP[3])
        F_concatenated = torch.cat((F_fAOP_4, F_fI0_4), dim=1)
        fuse_4 = self.conv[3](F_concatenated)
        #print(F_fAOP_3.shape,12589)
        return [fuse_1, fuse_2, fuse_3, fuse_4 ,F_fAOP_3[0] ,F_fI0_3[0]]

        """
        #F_fAOP_1, F_fI0_1 = self.CMDAF_1(I0[0], AOLP[0])
        F_concatenated = torch.cat((I0[0], AOLP[0]), dim=1)
        fuse_1 = self.conv[0](F_concatenated)
        #F_fAOP_2, F_fI0_2 = self.CMDAF_2(I0[1], AOLP[1])
        F_concatenated = torch.cat((I0[1], AOLP[1]), dim=1)
        fuse_2 = self.conv[1](F_concatenated)
        #F_fAOP_3, F_fI0_3 = self.CMDAF_3(I0[2], AOLP[2])
        F_concatenated = torch.cat((I0[2], AOLP[2]), dim=1)
        fuse_3 = self.conv[2](F_concatenated)
        #F_fAOP_4, F_fI0_4 = self.CMDAF_4(I0[3], AOLP[3])
        F_concatenated = torch.cat((I0[3], AOLP[3]), dim=1)
        fuse_4 = self.conv[3](F_concatenated)
        return [fuse_1, fuse_2, fuse_3, fuse_4]
        """
        """
        F_fAOP_1, F_fI0_1= self.CMDAF(I0[0], AOLP[0])
        F_concatenated = torch.cat((F_fAOP_1, F_fI0_1), dim=1)
        fuse_1 = self.conv[0](F_concatenated)
        F_fAOP_2, F_fI0_2 = self.CMDAF(I0[1], AOLP[1])
        F_concatenated = torch.cat((F_fAOP_2, F_fI0_2), dim=1)
        fuse_2 = self.conv[1](F_concatenated)
        F_fAOP_3, F_fI0_3= self.CMDAF(I0[2], AOLP[2])
        F_concatenated = torch.cat((F_fAOP_3, F_fI0_3), dim=1)
        fuse_3 = self.conv[2](F_concatenated)
        F_fAOP_4, F_fI0_4 = self.CMDAF(I0[3], AOLP[3])
        F_concatenated = torch.cat((F_fAOP_4, F_fI0_4), dim=1)
        fuse_4 = self.conv[3](F_concatenated)
        return [fuse_1, fuse_2, fuse_3, fuse_4 ,F_fAOP_3[0] ,F_fI0_3[0]]
        """
    def CMDAF_1(self,I0, AOP):
        sub_I0_AOP = I0 - AOP
        w_I0_AOP = self.atten_1(sub_I0_AOP)

        sub_AOP_I0 = AOP - I0
        w_AOP_I0=self.atten_1(sub_AOP_I0)

        F_fI0 = I0 + w_AOP_I0 * sub_AOP_I0
        F_fAOP = AOP + w_I0_AOP * sub_I0_AOP
        return F_fI0 ,F_fAOP

    def CMDAF_2(self,I0, AOP):
        sub_I0_AOP = I0 - AOP
        w_I0_AOP = self.atten_2(sub_I0_AOP)

        sub_AOP_I0 = AOP - I0
        w_AOP_I0=self.atten_2(sub_AOP_I0)

        F_fI0 = I0 + w_AOP_I0 * sub_AOP_I0
        F_fAOP = AOP + w_I0_AOP * sub_I0_AOP
        return F_fI0 ,F_fAOP
    def CMDAF_3(self,I0, AOP):
        sub_I0_AOP = I0 - AOP
        w_I0_AOP = self.atten_3(sub_I0_AOP)

        sub_AOP_I0 = AOP - I0
        w_AOP_I0=self.atten_3(sub_AOP_I0)

        F_fI0 = I0 + w_AOP_I0 * sub_AOP_I0
        F_fAOP = AOP + w_I0_AOP * sub_I0_AOP
        return F_fI0 ,F_fAOP
    def CMDAF_4(self,I0, AOP):
        sub_I0_AOP = I0 - AOP
        w_I0_AOP = self.atten_4(sub_I0_AOP)

        sub_AOP_I0 = AOP - I0
        w_AOP_I0=self.atten_4(sub_AOP_I0)

        F_fI0 = I0 + w_AOP_I0 * sub_AOP_I0
        F_fAOP = AOP + w_I0_AOP * sub_I0_AOP
        return F_fI0 ,F_fAOP


    def CMDAF(self, I0, AOP):
        #img_Imask_glant = F.interpolate(img_Imask_glant, size=I0.shape[2:], mode='bilinear', align_corners=False)

        sub_I0_AOP = I0 - AOP
        sub_w_I0_AOP = torch.mean(sub_I0_AOP, dim=[1, 2], keepdim=True)  # Global Average Pooling
        #w_I0_AOP=self.atten_1(sub_I0_AOP)
        w_I0_AOP = torch.sigmoid(sub_w_I0_AOP)

        sub_AOP_I0 = AOP - I0
        sub_w_AOP_I0 = torch.mean(sub_AOP_I0, dim=[1, 2], keepdim=True)  # Global Average Pooling
        w_AOP_I0 = torch.sigmoid(sub_w_AOP_I0)
        """
        sub_DOLP_AOP = DOLP - AOP
        #sub_DOLP_AOP = sub_DOLP_AOP * (1 - img_Imask_glant)
        sub_w_DOLP_AOP = torch.mean(sub_DOLP_AOP, dim=[1, 2], keepdim=True)  # Global Average Pooling
        w_DOLP_AOP = torch.sigmoid(sub_w_DOLP_AOP)

        sub_AOP_DOLP = AOP - DOLP
        #sub_AOP_DOLP = sub_AOP_DOLP * (1 - img_Imask_glant)
        sub_w_AOP_DOLP = torch.mean(sub_AOP_DOLP, dim=[1, 2], keepdim=True)  # Global Average Pooling
        w_AOP_DOLP = torch.sigmoid(sub_w_AOP_DOLP)

        sub_I0_DOLP = I0 - DOLP
        #sub_I0_DOLP = sub_I0_DOLP * (1 - img_Imask_glant)
        sub_w_I0_DOLP = torch.mean(sub_I0_DOLP, dim=[1, 2], keepdim=True)  # Global Average Pooling
        w_I0_DOLP = torch.sigmoid(sub_w_I0_DOLP)

        sub_DOLP_I0 = DOLP - I0
        #sub_DOLP_I0 = sub_DOLP_I0 * (1 - img_Imask_glant)
        sub_w_DOLP_I0 = torch.mean(sub_DOLP_I0, dim=[1, 2], keepdim=True)  # Global Average Pooling
        w_DOLP_I0 = torch.sigmoid(sub_w_DOLP_I0)
        """
        # F_dI0 = w_I0_AOP  * sub_I0_AOP  # 放大差分信号，此处是否应该调整为sub_ir_vi
        # F_dAOP = w_AOP_I0 * sub_AOP_I0

        F_fI0 = I0 + w_AOP_I0 * sub_AOP_I0
        F_fAOP = AOP + w_I0_AOP * sub_I0_AOP
        #F_fDOLP = DOLP * (1 - img_Imask_glant) + w_I0_DOLP * sub_I0_DOLP + w_AOP_DOLP * sub_AOP_DOLP##jia mask
        #F_fDOLP = DOLP  + w_I0_DOLP * sub_I0_DOLP + w_AOP_DOLP * sub_AOP_DOLP##bujia mask

        return  F_fI0,F_fAOP

    def attention_module(self,Fd):
        w_avg = self.ca_avg(x)
        w_max = self.ca_max(x)
        w = torch.cat([w_avg, w_max], dim=1)
        w = self.conv1(w)
        # SA
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x_sa = torch.cat([avgout, maxout], dim=1)
        x1 = self.conv(x_sa)
        attention_map = self.sigmod(w * x1)
        return attention_map

def vgg_preprocess(batch):
    #tensortype = type(batch.data)
    if batch.size(1) < 3:
        batch = torch.cat([batch] * 3, dim=1)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    #batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    batch = batch  * 255
    """
    if opt.vgg_mean:
        mean = tensortype(batch.data.size())
        mean[:, 0, :, :] = 103.939
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        batch = batch.sub(Variable(mean)) # subtract mean
    """
    return batch

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        #self.opt = opt
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        """
        if self.opt.no_vgg_instance:
            return torch.mean((img_fea - target_fea) ** 2)
        else:
        """
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

def load_vgg16(model_dir, gpu_ids):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
    #     if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
    #         os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
    #     vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
    #     vgg = Vgg16()
    #     for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
    #         dst.data[:] = src
    #     torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    # vgg.cuda()
    vgg.cuda(device=gpu_ids[0])
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    vgg = torch.nn.DataParallel(vgg, gpu_ids)
    return vgg


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        #if opt.vgg_choose != "no_maxpool":
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        relu4_1 = h
        h = F.relu(self.conv4_2(h), inplace=True)
        relu4_2 = h
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)
        relu4_3 = h
        """
        if opt.vgg_choose != "no_maxpool":
            if opt.vgg_maxpooling:
                h = F.max_pool2d(h, kernel_size=2, stride=2)
        """
        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        relu5_2 = F.relu(self.conv5_2(relu5_1), inplace=True)
        conv5_3 = self.conv5_3(relu5_2)
        h = F.relu(conv5_3, inplace=True)
        relu5_3 = h
        """
        if opt.vgg_choose == "conv4_3":
            return conv4_3
        elif opt.vgg_choose == "relu4_2":
            return relu4_2
        elif opt.vgg_choose == "relu4_1":
            return relu4_1
        elif opt.vgg_choose == "relu4_3":
            return relu4_3
        elif opt.vgg_choose == "conv5_3":
            return conv5_3
        elif opt.vgg_choose == "relu5_1":
            return relu5_1
        elif opt.vgg_choose == "relu5_2":
            return relu5_2
        elif opt.vgg_choose == "relu5_3" or "maxpool":
        """
        return relu5_3