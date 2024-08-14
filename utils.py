import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from args_fusion import args
#from scipy.misc import imread, imsave, imresize
import matplotlib as mpl
import cv2
import glob
from os import listdir
from os.path import join

def load_images(file):
    #im = Image.open(file)
    im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = np.array(im, dtype="float32") / 255.0
    img_norm = np.float32(img)
    return img_norm

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)#调用返回指定目录下的文件和子目录列表。
    dir.sort()#对获取的文件列表进行排序，这里使用的是默认的字母顺序排序。
    for file in dir:
        name = file.lower()#将当前文件名转换为小写字母，并将结果存储在变量 name 中。
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))#将完整的文件路径添加到 images 列表中
        name1 = name.split('.')#将文件名添加到 names 列表中，不包括文件扩展名。
        names.append(name1[0])#函数返回两个列表：images 包含所有图像文件的完整路径，names 包含所有图像文件的名称（不包括扩展名）
    return images

def list_images_polarization(train_path):
    # load_data
    train_I0_data = []
    train_I45_data = []
    train_I90_data = []
    train_I135_data = []
    train_I0 = glob.glob(train_path + '/I0/*.jpg')
    train_I45 = glob.glob(train_path + '/I45/*.jpg')
    train_I90 = glob.glob(train_path + '/I90/*.jpg')
    train_I135 = glob.glob(train_path + '/I135/*.jpg')

    train_I0.sort()  # 对文件名列表进行排序，以确保加载图像的顺序与文件名列表的顺序一致。
    train_I45.sort()
    train_I90.sort()
    train_I90.sort()
    print('[*] Number of training data_I0/I45/I90/I135: %d' % len(train_I0))
    for idx in range(len(train_I0)):
        image_I0 = load_images(train_I0[idx])
        train_I0_data.append(image_I0)
        image_I45 = load_images(train_I45[idx])
        train_I45_data.append(image_I45)
        image_I90 = load_images(train_I90[idx])
        train_I90_data.append(image_I90)
        image_I135 = load_images(train_I135[idx])
        train_I135_data.append(image_I135)##返回的就是归一化后的
    return train_I0_data,train_I45_data,train_I90_data,train_I135_data

def DLOP_calculate(I0,I45,I90,I135):
    #DOLP_list = []
    #for i in range(len(I0)):
    S0 = (I0 + I45+ I90 + I135) / 2
    S1 = I0 - I90
    S2 = I45 - I135
    DOLP = np.sqrt(S1 ** 2 + S2 ** 2) / (S0+0.001)
        #DOLP = torch.sqrt(S1 ** 2 + S2 ** 2) / S0
        #DOLP_list.append(DOLP)
    return DOLP

def Isp_calculate(I0,I45,I90,I135):
    #Isp_min_list = []
    #Isp_max_list=[]
    #for i in range(len(I0)):
    images = [I0, I45, I90, I135]
    Isp_counts = [sum(image >= 1) for image in images]

        # 找出饱和像素数量最少的图像的索引
    min_index = Isp_counts.index(min(Isp_counts))
    Isp_min=images[min_index]

        # 找出饱和像素数量最多的图像的索引
    max_index = Isp_counts.index(max(Isp_counts))
    Isp_max=images[max_index]

        # 返回结果
    return Isp_min, Isp_max


def Imask_calculate(I0,I45,I90,I135):
    Imask_data = []
    for i in range(len(I0)):
        height, width = I0[i].shape[-2:]  # 获取高度和宽度
        #Imask = np.ones((height, width), dtype=np.uint8)  # 初始化 Imask 全为 1
        Imask = torch.ones((1, height, width), dtype=torch.uint8).cuda()
        count=0
        # 循环遍历每个像素位置
        for h in range(height):
            for w in range(width):
                # 如果任何一个偏振图像中该位置的像素值为255，则将 Imask 中对应位置的像素值设为0
                #if I0[i][h, w] == 1 or I45[i][h, w] == 1 or I90[i][h, w] == 1 or I135[i][h, w] == 1:
                if torch.any(I0[i][0, h, w] == 1) or torch.any(I45[i][0, h, w] == 1) or torch.any(
                        I90[i][0, h, w] == 1) or torch.any(I135[i][0, h, w] == 1):
                    Imask[0, h, w] = 0
                    count+=1
        #print("count",count)
        Imask_data.append(Imask)

    return Imask_data

# 计算标准差图像
def calculate_std_image(image, radius=5):
    std_image = cv2.GaussianBlur(image.astype(np.float32), (0, 0), radius)
    std_image = np.sqrt(cv2.subtract(std_image ** 2, cv2.GaussianBlur(image.astype(np.float32) ** 2, (0, 0), radius)))
    return std_image

# 将标准差图像转换为二值图像
def convert_to_binary(std_image, threshold=0.03):
    binary_image = np.zeros_like(std_image, dtype=np.uint8)
    binary_image[std_image > threshold] = 1
    return binary_image
def Imask_calculate(Isp_max):
    Isp_mask_sp = np.where(Isp_max == 1., 1., 0)
    # 计算标准差图像
    std_image = calculate_std_image(Isp_max)

    # 转换为二值图像
    binary_image = convert_to_binary(std_image)


    # 计算连通区域
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # 获取饱和像素位置
    saturated_positions = set(np.unique(labels[Isp_max > 0]))

    for label in range(1, np.max(labels) + 1):
        if label not in saturated_positions:
            labels[labels == label] = 0
    return Isp_mask_sp ,labels

"""
# 定义 Sobel 算子的卷积核
sobel_x_kernel = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

sobel_y_kernel = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

def Iweight_calculate_process(I0):
    # 计算 Sobel 算子对图像的梯度
    I0_tensor = torch.as_tensor(I0, dtype=torch.float32).cuda()
    I0_tensor = torch.unsqueeze(I0_tensor, dim=0)  # 添加 batch_size 维度
    #I0_tensor = torch.unsqueeze(I0_tensor, dim=0)  # 添加 in_channels 维度
    sobel_x_output = F.conv2d(I0_tensor, sobel_x_kernel, padding=1)
    sobel_y_output = F.conv2d(I0_tensor, sobel_y_kernel, padding=1)
    grad_magnitude = torch.sqrt(sobel_x_output ** 2 + sobel_y_output ** 2)

    # 大于 0.02 的像素标记为 1，其余的为 0
    mask = (grad_magnitude > 0.02).float()

    #print("mask",mask.shape)

    # 计算连通区域
   # _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.cpu().numpy().astype(np.uint8), connectivity=8)
    # 将 PyTorch 张量移动到 CPU，并转换为 NumPy 数组
    mask_np = mask.cpu().numpy()

    # 从数组中移除单维度（如果有的话）
    mask_np = np.squeeze(mask_np)

    # 将数据转换为 uint8 类型
    mask_np = mask_np.astype(np.uint8)

    # 应用 connected components 算法
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

    # 找到像素为 1 的连通区域中面积最大的那个
    max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # 保留最大的连通区域，其余的设为 0
    I1_mask = np.where(labels == max_label, 1, 0)
    I1_mask = np.expand_dims(I1_mask, axis=0)
    # 对 I1 进行形态学膨胀，半径为 5
    kernel = np.ones((5, 5), np.uint8)
    Iweight = cv2.dilate(I1_mask.astype(np.uint8), kernel, iterations=1)

    return Iweight

def Iweight_calculate(I0):
    train_Iweight_data = []

    for i in range(len(I0)):
        Iweight = Iweight_calculate_process(I0[i])
        train_Iweight_data.append(Iweight)
    return train_Iweight_data
"""
"""
def Iweight_calculate(I0):
    Iweight_data = []
    kernel = np.ones((5, 5), np.uint8)
    #kernel = torch.ones((5, 5)).cuda()
    for i in range(len(I0)):
        # 计算梯度
        Sobel_x = cv2.Sobel(I0[i], cv2.CV_64F, 1, 0, ksize=3)
        Sobel_y = cv2.Sobel(I0[i], cv2.CV_64F, 0, 1, ksize=3)
        gradient = cv2.magnitude(Sobel_x, Sobel_y)

        # 标记大于0.02的像素为1，其余为0
        Gamg = np.zeros_like(gradient, dtype=np.uint8)
        Gamg[gradient > 0.03] = 1

        # 计算连通区域
        _, labels, stats, _ = cv2.connectedComponentsWithStats(Gamg, connectivity=8)

        # 获取像素值为1的区域中面积最大的连通区域
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        I1 = np.zeros_like(labels, dtype=np.uint8)
        I1[labels == largest_label] = 1

        # 对I1的像素值为1的区域进行形态学膨胀，半径为5
        I1 = cv2.dilate(I1, kernel, iterations=1)

        Iweight_data.append(I1)

    return Iweight_data
"""

def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        # img = tensor.clone().cpu().clamp(0, 255).numpy()
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        # img = tensor.clone().clamp(0, 255).numpy()
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U,D,V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        image_list = glob.glob(image_path + '/' + "*")
        num_imgs = len(image_list)
    original_image_list = image_list[:num_imgs]
    # random
    #random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    #print('BATCH SIZE %d.' % BATCH_SIZE)
    #print('Train images number %d.' % num_imgs)
    #print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_image_list = original_image_list[:-mod]
    batches = int(len(original_image_list) // BATCH_SIZE)
    return  batches


def get_image(path, height=256, width=256, flag=False):
    if flag is True:
        #image = imread(path, mode='RGB')
        image = cv2.imread(path)
    else:
        #image = imread(path, mode='L')
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = np.array(image, dtype="float32") / 255.0
        image = np.float32(image)

    if height is not None and width is not None:
        #image = imresize(image, [height, width], interp='nearest')
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return image


# load images - test phase
def get_test_image(paths, height=None, width=None, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        #image = imread(path, mode='L')
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = np.array(image, dtype="float32") / 255.0
        image = np.float32(image)
        if height is not None and width is not None:
            #image = imresize(image, [height, width], interp='nearest')
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

        base_size = 616
        h = image.shape[0]
        w = image.shape[1]

        c = 1

        if h > base_size or w > base_size:
            c = 4
            images = get_img_parts(image, h, w)###将图像分割为四部分
        else:

            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
            images.append(image)
            images = np.stack(images, axis=0)
            images = torch.from_numpy(images).float()

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images, h, w, c

"""
def get_img_parts(image, h, w):
    images = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    img1 = image[0:h_cen + 3, 0: w_cen + 3]
    img1 = np.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:h_cen + 3, w_cen - 2: w]
    img2 = np.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[h_cen - 2:h, 0: w_cen + 3]
    img3 = np.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[h_cen - 2:h, w_cen - 2: w]
    img4 = np.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    images.append(torch.from_numpy(img1).float())
    images.append(torch.from_numpy(img2).float())
    images.append(torch.from_numpy(img3).float())
    images.append(torch.from_numpy(img4).float())
    return images


def recons_fusion_images(img_lists, h, w):
    img_f_list = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    ones_temp = torch.ones(1, 1, h, w).cuda()
    for i in range(len(img_lists[0])):
        # img1, img2, img3, img4
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        # save_image_test(img1, './outputs/test/block1.png')
        # save_image_test(img2, './outputs/test/block2.png')
        # save_image_test(img3, './outputs/test/block3.png')
        # save_image_test(img4, './outputs/test/block4.png')

        img_f = torch.zeros(1, 1, h, w).cuda()
        count = torch.zeros(1, 1, h, w).cuda()

        img_f[:, :, 0:h_cen + 3, 0: w_cen + 3] += img1
        count[:, :, 0:h_cen + 3, 0: w_cen + 3] += ones_temp[:, :, 0:h_cen + 3, 0: w_cen + 3]
        img_f[:, :, 0:h_cen + 3, w_cen - 2: w] += img2
        count[:, :, 0:h_cen + 3, w_cen - 2: w] += ones_temp[:, :, 0:h_cen + 3, w_cen - 2: w]
        img_f[:, :, h_cen - 2:h, 0: w_cen + 3] += img3
        count[:, :, h_cen - 2:h, 0: w_cen + 3] += ones_temp[:, :, h_cen - 2:h, 0: w_cen + 3]
        img_f[:, :, h_cen - 2:h, w_cen - 2: w] += img4
        count[:, :, h_cen - 2:h, w_cen - 2: w] += ones_temp[:, :, h_cen - 2:h, w_cen - 2: w]
        img_f = img_f / count
        img_f_list.append(img_f)
    return img_f_list
"""

def get_img_parts(image, h, w):
    images = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    img1 = image[0:h_cen +0, 0: w_cen + 4]
    img1 = np.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:h_cen + 0, w_cen - 4: w]
    img2 = np.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[h_cen - 0:h, 0: w_cen + 4]
    img3 = np.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[h_cen - 0:h, w_cen - 4: w]
    img4 = np.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    images.append(torch.from_numpy(img1).float())
    images.append(torch.from_numpy(img2).float())
    images.append(torch.from_numpy(img3).float())
    images.append(torch.from_numpy(img4).float())
    return images


def recons_fusion_images(img_lists, h, w):
    img_f_list = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    ones_temp = torch.ones(1, 1, h, w).cuda()
    for i in range(len(img_lists[0])):
        # img1, img2, img3, img4
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        # save_image_test(img1, './outputs/test/block1.png')
        # save_image_test(img2, './outputs/test/block2.png')
        # save_image_test(img3, './outputs/test/block3.png')
        # save_image_test(img4, './outputs/test/block4.png')

        img_f = torch.zeros(1, 1, h, w).cuda()
        count = torch.zeros(1, 1, h, w).cuda()

        img_f[:, :, 0:h_cen + 0, 0: w_cen + 4] += img1
        count[:, :, 0:h_cen + 0, 0: w_cen + 4] += ones_temp[:, :, 0:h_cen + 0, 0: w_cen + 4]
        img_f[:, :, 0:h_cen + 0, w_cen - 4: w] += img2
        count[:, :, 0:h_cen + 0, w_cen - 4: w] += ones_temp[:, :, 0:h_cen + 0, w_cen - 4: w]
        img_f[:, :, h_cen - 0:h, 0: w_cen + 4] += img3
        count[:, :, h_cen - 0:h, 0: w_cen + 4] += ones_temp[:, :, h_cen - 0:h, 0: w_cen + 4]
        img_f[:, :, h_cen - 0:h, w_cen - 4: w] += img4
        count[:, :, h_cen - 0:h, w_cen - 4: w] += ones_temp[:, :, h_cen - 0:h, w_cen - 4: w]
        img_f = img_f / count
        img_f_list.append(img_f)
    return img_f_list

def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
        # img_fusion = img_fusion.cpu().clamp(0, 255).data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()
    ####
    """
    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    """
    ####

    image_numpy = (np.transpose(img_fusion, (1, 2, 0))) * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    #    print(img_fusion.shape[2],111)
    image_numpy = image_numpy.astype(np.uint8)
    image_numpy = image_numpy.reshape([image_numpy.shape[0], image_numpy.shape[1]])
    image_pil = Image.fromarray(image_numpy, mode='L')
    #image_pil = Image.fromarray(image_numpy)
    #image_pil = image_pil.convert("L")
    image_pil.save(output_path)
    #img_fusion=image_numpy
    #####
    """
    # cv2.imwrite(output_path, img_fusion)
    if img_fusion.shape[2] == 1:
        print(img_fusion.shape[2],111)
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    # 	img_fusion = imresize(img_fusion, [h, w])
    #imsave(output_path, img_fusion)
    cv2.imwrite(output_path, img_fusion)
    """
    #####

def get_train_images(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images_ir = []
    images_vi = []
    for path in paths:
        image = get_image(path, height, width, flag)
        image = np.reshape(image, [1, height, width])
        # imsave('./outputs/ir_gray.jpg', image)
        # image = image.transpose(2, 0, 1)
        images_ir.append(image)

        path_vi = path.replace('lwir', 'visible')
        image = get_image(path_vi, height, width, flag)
        image = np.reshape(image, [1, height, width])
        # imsave('./outputs/vi_gray.jpg', image)
        # image = image.transpose(2, 0, 1)
        images_vi.append(image)

    images_ir = np.stack(images_ir, axis=0)
    images_ir = torch.from_numpy(images_ir).float()

    images_vi = np.stack(images_vi, axis=0)
    images_vi = torch.from_numpy(images_vi).float()
    return images_ir, images_vi


def get_train_images_auto(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:##flag=False
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


# 自定义colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)




