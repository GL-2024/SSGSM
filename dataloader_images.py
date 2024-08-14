import os
import sys

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import glob
import random
import cv2
from args_fusion import args

random.seed(1143)
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def transform_matrix_offset_center(matrix, x, y):
    """Return transform matrix offset center.

	Parameters
	----------
	matrix : numpy array
		Transform matrix
	x, y : int
		Size of image.

	Examples
	--------
	- See ``rotation``, ``shear``, ``zoom``.
	"""
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.
	Args:
		img (ndarray): Image to be rotated.
		angle (float): Rotation angle in degrees. Positive values mean
			counter-clockwise rotation.
		center (tuple[int]): Rotation center. If the center is None,
			initialize it as the center of the image. Default: None.
		scale (float): Isotropic scale factor. Default: 1.0.
	"""
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
                                 borderValue=(0, 0, 0), )
    return rotated_img


def zoom(x, zx, zy, row_axis=0, col_axis=1):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]

    matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = cv2.warpAffine(x, matrix[:2, :], (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
                       borderValue=(0, 0, 0), )
    return x


def augmentation(img1, img2):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    # rot90=random.random() < 0.5
    rot = random.random() < 0.3
    zo = random.random() < 0.3
    angle = random.random() * 180 - 90
    if hflip:
        img1 = cv2.flip(img1, 1)
        img2 = cv2.flip(img2, 1)
    if vflip:
        img1 = cv2.flip(img1, 0)
        img2 = cv2.flip(img2, 0)
    # if rot90:
    # img1 = img1.transpose(1, 0, 2)
    # img2 = img2.transpose(1,0,2)
    if zo:
        zoom_range = (0.7, 1.3)
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        img1 = zoom(img1, zx, zy)
        img2 = zoom(img2, zx, zy)
    if rot:
        img1 = img_rotate(img1, angle)
        img2 = img_rotate(img2, angle)
    return img1, img2


def preprocess_aug(img1, img2):
    img1 = np.uint8((np.asarray(img1)))
    img2 = np.uint8((np.asarray(img2)))
    # img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    # img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    img1, img2 = augmentation(img1, img2)
    # img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    return img1, img2


def populate_train_list(images_path):
    image_list = glob.glob(images_path + '/' + "*")
    train_list = sorted(image_list)
    return train_list


def random_order(train_list_I0, train_list_I45, train_list_I90, train_list_I135):
    zipped_lists = list(zip(train_list_I0, train_list_I45, train_list_I90, train_list_I135))
    # 对打包后的列表进行打乱
    random.shuffle(zipped_lists)
    # 解包打乱后的列表，恢复成原来的四个列表
    train_list_I0_shuffled, train_list_I45_shuffled, train_list_I90_shuffled, train_list_I135_shuffled = zip(
        *zipped_lists)
    return train_list_I0_shuffled, train_list_I45_shuffled, train_list_I90_shuffled, train_list_I135_shuffled


def guided_filter(I, p, radius=5, eps=0.04):
    """
    Guided filter implementation for PyTorch.
    Reference: https://arxiv.org/abs/1505.00996

    Args:
        I (Tensor): Guidance image (should be a grayscale image).
        p (Tensor): Input image (should have the same shape as the guidance image).
        radius (int): Radius of the filter.
        eps (float): Regularization parameter.

    Returns:
        Tensor: Filtered output.
    """
    # Convert tensors to numpy arrays
    I, p = I.numpy(), p.numpy()


    # Preprocess I and p
    mean_I = convolve(I, np.ones(( radius, radius)) / (radius ** 2), mode='reflect')
    mean_p = convolve(p, np.ones(( radius, radius)) / (radius ** 2), mode='reflect')
    corr_I = convolve(I * I, np.ones((radius, radius)), mode='reflect')
    corr_Ip = convolve(I * p, np.ones(( radius, radius)), mode='reflect')

    # Compute covariance matrix
    cov_Ip = corr_Ip - mean_I * mean_p
    var_I = corr_I - mean_I * mean_I

    # Compute a and b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Postprocess a and b
    mean_a = convolve(a, np.ones(( radius, radius)) / (radius ** 2), mode='reflect')
    mean_b = convolve(b, np.ones(( radius, radius)) / (radius ** 2), mode='reflect')

    # Compute output
    q = mean_a * I + mean_b

    return torch.from_numpy(q).float()


def DLOP_calculate(I0, I45, I90, I135):
    DOLP_list = []
    AOP_list = []
    for i in range(len(I0)):
        S0 = (I0[i] + I45[i] + I90[i] + I135[i]) / 2
        S1 = I0[i] - I90[i]
        S2 = I45[i] - I135[i]
        # DOLP = np.sqrt(S1 ** 2 + S2 ** 2)/S0
        DOLP = torch.sqrt(S1 ** 2 + S2 ** 2) / (S0+ 1e-10)
        AOP = 0.5 * (np.arctan2(S2, S1 + 1e-10)) * 180 / np.pi
        condition = AOP < 0
        AOP[condition] += 180
        AOP2=AOP/180
        # 使用示例
        #DOLP_filtered = guided_filter(DOLP, DOLP, radius=3, eps=0.02).squeeze()
        #AOP_filtered = guided_filter(AOP2, AOP2, radius=3, eps=0.02).squeeze()
        """
        AOP2_show = AOP.squeeze().cpu().numpy()
        # 将数据转换为 uint8 类型
        binary_image_np = (AOP2_show * 255).astype(np.uint8)

        # 显示图像
        cv2.imshow('Binary Image', binary_image_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        DOLP_list.append(DOLP)
        AOP_list.append(AOP2)
    return DOLP_list, AOP_list


def Isp_calculate(I0, I45, I90, I135):
    Isp_min_list = []
    Isp_max_list = []
    for i in range(len(I0)):
        images = [I0[i], I45[i], I90[i], I135[i]]
        Isp_counts = [torch.sum(image >= 1) for image in images]

        # 找出饱和像素数量最少的图像的索引
        min_index = Isp_counts.index(min(Isp_counts))
        image1=images[min_index]
        #image1=image1/7
        """
        AOP2_show = image1.squeeze().cpu().numpy()
        # 将数据转换为 uint8 类型
        binary_image_np = (AOP2_show * 255).astype(np.uint8)

        # 显示图像
        cv2.imshow('Binary Image', binary_image_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        Isp_min_list.append(image1)

        # 找出饱和像素数量最多的图像的索引
        max_index = Isp_counts.index(max(Isp_counts))
        Isp_max_list.append(images[max_index])

    # 返回结果
    return Isp_min_list, Isp_max_list


def Imask_glant_calculate(Isp_max):
    Isp_max = Isp_max.cpu().numpy().astype(np.float32)
    Isp_max = Isp_max.squeeze()
    # 计算标准差图像
    radius = 3
    blur_squared = cv2.blur(Isp_max ** 2, (radius * 2 + 1, radius * 2 + 1))
    G = cv2.blur(Isp_max ** 2, (radius * 2 + 1, radius * 2 + 1)) - cv2.blur(Isp_max,
                                                                            (radius * 2 + 1, radius * 2 + 1)) ** 2
    G = np.sqrt(np.maximum(G, 0))
    # 大于 0.02 的像素标记为 1，其余的为 0
    mask = np.where(G > 0.02, 1., 0.)

    num_regions, label_image = cv2.connectedComponents(mask.astype(np.uint8))
    # 获取Isp_max图像中饱和像素的位置
    saturated_pixels = np.where(Isp_max == 1)

    # 将原Isp_max图像中饱和像素位置所在的连通区域保留，其余的连通区域置为0
    for label in range(1, num_regions):
        region_pixels = np.where(label_image == label)
        if np.any(np.isin(region_pixels, saturated_pixels)):
            mask[label_image == label] = 1
        else:
            mask[label_image == label] = 0
    # mask=mask.astype(np.float32)
    mask = torch.from_numpy(mask).float()

    return mask


def Imask_calculate(Isp_max):
    #train_Imask_sp_data = []
    train_Imask_glant_data = []
    for i in range(len(Isp_max)):
        # 计算二值图像
        binary_image = torch.where(Isp_max[i] == 1., 1, 0).float()
        #train_Imask_sp_data.append(binary_image)  ###饱和像素为1

        Imask_glant = Imask_glant_calculate(Isp_max)  ###耀光区域为1

        train_Imask_glant_data.append(Imask_glant)
    return  train_Imask_glant_data


class lowlight_loader(data.Dataset):

    def __init__(self, train_I0_imgs_path, train_I45_imgs_path, train_I90_imgs_path, train_I135_imgs_path, height,
                 width):
        self.train_list_I0 = populate_train_list(train_I0_imgs_path)
        self.train_list_I45 = populate_train_list(train_I45_imgs_path)
        self.train_list_I90 = populate_train_list(train_I90_imgs_path)
        self.train_list_I135 = populate_train_list(train_I135_imgs_path)
        self.train_list_I0_shuffled, self.train_list_I45_shuffled, self.train_list_I90_shuffled, self.train_list_I135_shuffled = random_order(
            self.train_list_I0, self.train_list_I45, self.train_list_I90, self.train_list_I135)

        self.height = height
        self.width = width

        # print("Total training examples :", len(self.train_list_I0))
        num_imgs = len(self.train_list_I0)
        batches = num_imgs // args.batch_size_second
        print('Total training examples %d.' % num_imgs)
        print('Train images batches %d.' % batches)

    def __getitem__(self, index):
        data_I0_path = self.train_list_I0_shuffled[index]
        data_I45_path = self.train_list_I45_shuffled[index]
        data_I90_path = self.train_list_I90_shuffled[index]
        data_I135_path = self.train_list_I135_shuffled[index]

        data_I0 = cv2.imread(data_I0_path, cv2.IMREAD_GRAYSCALE)
        data_I45 = cv2.imread(data_I45_path, cv2.IMREAD_GRAYSCALE)
        data_I90 = cv2.imread(data_I90_path, cv2.IMREAD_GRAYSCALE)
        data_I135 = cv2.imread(data_I135_path, cv2.IMREAD_GRAYSCALE)

        data_I0 = np.array(data_I0, dtype="float32") / 255.0
        data_I0 = np.float32(data_I0)
        data_I45 = np.array(data_I45, dtype="float32") / 255.0
        data_I45 = np.float32(data_I45)
        data_I90 = np.array(data_I90, dtype="float32") / 255.0
        data_I90 = np.float32(data_I90)
        data_I135 = np.array(data_I135, dtype="float32") / 255.0
        data_I135 = np.float32(data_I135)

        data_I0 = cv2.resize(data_I0, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        data_I45 = cv2.resize(data_I45, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        data_I90 = cv2.resize(data_I90, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        data_I135 = cv2.resize(data_I135, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        # nor_image, _ = preprocess_aug(nor_image, nor_image)###随机水平翻转、垂直翻转、旋转、缩放和旋转
        data_I0 = torch.from_numpy(data_I0).float().unsqueeze(0)
        data_I45 = torch.from_numpy(data_I45).float().unsqueeze(0)
        data_I90 = torch.from_numpy(data_I90).float().unsqueeze(0)
        data_I135 = torch.from_numpy(data_I135).float().unsqueeze(0)

        img_DOLP, img_AOP = DLOP_calculate(data_I0, data_I45, data_I90, data_I135)
        img_DOLP = torch.stack(img_DOLP)
        img_AOP = torch.stack(img_AOP)

        Isp_min, Isp_max = Isp_calculate(data_I0, data_I45, data_I90, data_I135)
        img_Isp_min = torch.stack(Isp_min)
        img_Isp_max = torch.stack(Isp_max)

        img_Imask_glant = Imask_calculate(img_Isp_max)  ##耀光区域为1
        #img_Imask_sp = torch.stack(img_Imask_sp)
        img_Imask_glant = torch.stack(img_Imask_glant)
        # img_Imask_glant = torch.tensor(img_Imask_glant, dtype=torch.float32, device='cuda')

        return img_DOLP, img_AOP, img_Isp_min, img_Isp_max,  img_Imask_glant


    def __len__(self):
        return len(self.train_list_I0)


class lowlight_loader1(data.Dataset):

    def __init__(self, train_I0_imgs_path ,height,width):
        self.train_list_I0 = populate_train_list(train_I0_imgs_path)

        #self.train_list_I0_shuffled = random_order(self.train_list_I0)

        self.height = height
        self.width = width

        # print("Total training examples :", len(self.train_list_I0))
        num_imgs = len(self.train_list_I0)
        batches = num_imgs // args.batch_size_second
        print('Total training examples %d.' % num_imgs)
        print('Train images batches %d.' % batches)

    def __getitem__(self, index):
        data_I0_path = self.train_list_I0[index]


        data_I0 = cv2.imread(data_I0_path, cv2.IMREAD_GRAYSCALE)


        data_I0 = np.array(data_I0, dtype="float32") / 255.0
        data_I0 = np.float32(data_I0)


        data_I0 = cv2.resize(data_I0, (self.width, self.height), interpolation=cv2.INTER_NEAREST)


        # nor_image, _ = preprocess_aug(nor_image, nor_image)###随机水平翻转、垂直翻转、旋转、缩放和旋转
        data_I0 = torch.from_numpy(data_I0).float().unsqueeze(0)


        return data_I0

    def __len__(self):
        return len(self.train_list_I0)