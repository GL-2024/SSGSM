# -*- coding:utf-8 -*-
#@Project: NestFuse for image fusion
#@Author: Li Hui, Jiangnan University
#@Email: hui_li_jnu@163.com
#@File : test.py

import os
import torch
from torch.autograd import Variable
from net import NestFuse_autoencoder
import utils
from args_fusion import args
import numpy as np
import glob
import dataloader_images as dataloader_sharp
from net_tre_class import Decoder
import matplotlib.pyplot as plt

def load_model(path, deepsupervision,i):
	input_nc = 1
	output_nc = 1
	nb_filter = [64, 112, 160, 208, 256]
	if i==1:
		nest_model = Decoder(nb_filter, output_nc, deepsupervision)
		nest_model.load_state_dict(torch.load(path))
	else:
		nest_model = NestFuse_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)
		nest_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
	nest_model.cuda()

	return nest_model




def run_demo(model_AOP,model_DOLP,model_Isp_min,model_decoder, I0_path, I45_path,I90_path,I135_path, output_path_root, test_path,index):
	img_I0, h, w, c = utils.get_test_image(I0_path)
	img_I45, h, w, c = utils.get_test_image(I45_path)#,512,616
	img_I90, h, w, c = utils.get_test_image(I90_path)
	img_I135, h, w, c = utils.get_test_image(I135_path)
	print(img_I90[0].shape)
	# dim = img_ir.shape

	if c == 1:
		print(1)
		DOLP, AOP = dataloader_sharp.DLOP_calculate(img_I0, img_I45, img_I90, img_I135)
		DOLP = torch.stack(DOLP)
		AOP = torch.stack(AOP)
		Isp_min, Isp_max = dataloader_sharp.Isp_calculate(img_I0, img_I45, img_I90, img_I135)
		Isp_min = torch.stack(Isp_min)
		Isp_max = torch.stack(Isp_max)
		Isp_mask_glant = dataloader_sharp.Imask_calculate(Isp_max)
		# Isp_mask_sp = torch.stack(Isp_mask_sp)
		Isp_mask_glant = torch.stack(Isp_mask_glant)
		Isp_mask_glant = torch.tensor(Isp_mask_glant, dtype=torch.float32, device='cuda')
		if args.cuda:
			img_DOLP = DOLP.cuda()
			img_Isp_min = Isp_min.cuda()
			img_AOP = AOP.cuda()
			img_Isp_mask_glant = Isp_mask_glant.cuda()
		img_DOLP = Variable(img_DOLP, requires_grad=False)
		img_Isp_min = Variable(img_Isp_min, requires_grad=False)
		img_AOP = Variable(img_AOP, requires_grad=False)
		img_Isp_mask_glant = Variable(img_Isp_mask_glant, requires_grad=False)
		en_DOLP = model_DOLP.encoder(img_DOLP)
		#en_AOP = model_AOP.encoder(img_AOP)
		en_Isp_min = model_Isp_min.encoder(img_Isp_min)
		# fusion
		# f = nest_model.fusion(en_r, en_v, f_type)
		img_fusion_list ,AOP_feature ,I0_feature = model_decoder(en_Isp_min, en_DOLP)


	else:
		print(2)
		# fusion each block
		img_fusion_blocks = []
		for i in range(c):

			DOLP,AOP=dataloader_sharp.DLOP_calculate(img_I0[i],img_I45[i],img_I90[i],img_I135[i])
			DOLP = torch.stack(DOLP)
			AOP = torch.stack(AOP)
			Isp_min, Isp_max=dataloader_sharp.Isp_calculate(img_I0[i],img_I45[i],img_I90[i],img_I135[i])
			Isp_min = torch.stack(Isp_min)
			Isp_max = torch.stack(Isp_max)


			if args.cuda:
				img_DOLP = DOLP.cuda()
				img_Isp_min = Isp_min.cuda()
				img_AOP = AOP.cuda()
				#img_Isp_mask_glant = Isp_mask_glant.cuda()

			img_DOLP = Variable(img_DOLP, requires_grad=False)
			img_Isp_min = Variable(img_Isp_min, requires_grad=False)
			img_AOP = Variable(img_AOP, requires_grad=False)
			#img_Isp_mask_glant = Variable(img_Isp_mask_glant, requires_grad=False)

			en_DOLP = model_DOLP.encoder(img_DOLP)

			en_Isp_min= model_Isp_min.encoder(img_Isp_min)
			# fusion
			img_fusion_temp = model_decoder(en_Isp_min, en_DOLP)
			# decoder
			img_fusion_blocks.append(img_fusion_temp)
		img_fusion_list = utils.recons_fusion_images(img_fusion_blocks, h, w)

	############################ multi outputs ##############################################
	output_count = 0
	for img_fusion in img_fusion_list:
		image_path = I0_path.replace(test_path+ 'I0' + '/', output_path_root)
		file_name = '{:03d}.jpg'.format(index)
		output_path = output_path_root + file_name
		output_count += 1
		utils.save_image_test(img_fusion, output_path)
		print(output_path)


def main():
	# run demo
	test_path = "./images/verify_dataset/"##验证
	output_path="./images/result/1/"
	deepsupervision = False  # true for deeply supervision
	args.resume_decoder = './pretrain_model/model_propose/proposed_method.model'
	args.resume_AOP = './pretrain_model/AOPFinal_epoch_40_Mon_Apr_29_16_09_30_2024_1e2.model'  ###一阶段的
	args.resume_DOLP = './pretrain_model/Final_epoch_40_Mon_Apr_29_12_30_57_2024_1e2.model'
	args.resume_Isp_min = './pretrain_model/200_net_G_A.model'
	model_default=args.model_default
	with torch.no_grad():
		if deepsupervision:
			model_path = args.model_deepsuper
		else:
			model_decoder_path = args.resume_decoder
			model_AOP_path = args.resume_AOP
			model_DOLP_path = args.resume_DOLP
			model_Isp_min_path = args.resume_Isp_min
		model_AOP = load_model(model_AOP_path, deepsupervision,0)
		model_DOLP = load_model(model_DOLP_path, deepsupervision,0)
		model_Isp_min = load_model(model_Isp_min_path, deepsupervision,0)
		model_decoder = load_model(model_decoder_path, deepsupervision,1)
		print(model_decoder)

		if os.path.exists(output_path) is False:
			os.mkdir(output_path)
		print('Processing......  ')
		image_list_I0 = glob.glob(test_path + 'I0' + '/' + "*")
		image_list_I45 = glob.glob(test_path + 'I45' + '/' + "*")
		image_list_I90 = glob.glob(test_path + 'I90' + '/' + "*")
		image_list_I135 = glob.glob(test_path + 'I135' + '/' + "*")
		image_list_I0 = sorted(image_list_I0)
		image_list_I45 = sorted(image_list_I45)
		image_list_I90 = sorted(image_list_I90)
		image_list_I135 = sorted(image_list_I135)

		for i in range(len(image_list_I0)):
			index = i + 1

			I0_path=image_list_I0[i]
			I45_path = image_list_I45[i]
			I90_path = image_list_I90[i]
			I135_path = image_list_I135[i]
			print(I0_path)
			run_demo(model_AOP,model_DOLP,model_Isp_min,model_decoder, I0_path, I45_path,I90_path,I135_path, output_path, test_path,index)
	print('Done......')


if __name__ == '__main__':
	main()
