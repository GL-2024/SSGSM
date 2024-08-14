
class args():
	# training args
	GPU_num="0"
	epochs = 20   #"number of training epochs, default is 2"
	batch_size_once = 4  #"batch size for training, default is 4"
	batch_size_second = 4
	# the COCO dataset path in your computer
	# URL: http://images.cocodataset.org/zips/train2014.zip
	#dataset = "/data/Disk_B/MSCOCO2014/train2014/"

	##yijieduan
	#dataset = r"D:\\GengLin\EnlightenGAN-master\final_dataset/"

	##erjieduan
	dataset = r"D:\\GengLin\shujuzhengli\20240321\4.17/"
	verify_dataset="./images/verify_dataset/"
	#verify_dataset= r"D:\\GengLin\shujuzhengli\20240321\4.17/"
	verify_result_list_path = "./images/verify_result_dataset/"
	HEIGHT = 256
	WIDTH = 256

	save_model_dir_autoencoder = "models/nestfuse_autoencoder"
	save_model_dir_autoencoder_fuse = "./models/nestfuse_autoencoder"
	save_loss_dir = './models/loss_autoencoder/'
	save_loss_dir_fuse = './models/loss_autoencoder/'

	cuda = True
	ssim_weight = [1,10,100,1000,10000]
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

	lr = 1e-4  #"learning rate, default is 0.001"
	lr_light = 1e-4  # "learning rate, default is 0.001"
	log_interval = 1  #"number of images after which the training loss is logged, default is 500"
	resume = None

	# for test, model_default is the model used in paper
	model_default = './models/nestfuse_1e2.model'
	model_deepsuper = './models/nestfuse_1e2_deep_super.model'

	num_workers=False##多线程读取数据，大于0可加快数据加载速度，但是如果设置过大可能会导致内存不足或者系统负载过高。


