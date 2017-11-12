from easydict import EasyDict as edict
import json

config = edict()

config.model = edict()
config.model.result_path = "samples"
config.model.checkpoint_path = "checkpoint"
config.model.log_path = "log"
config.model.scale = 4
config.model.scale8 = 8
config.model.resblock_depth = 10
config.model.recursive_depth = 1

config.valid = edict()
config.valid.hr_folder_path = 'data2017/DIV2K_valid_HR/'
config.valid.lrx2_folder_path = 'data2017/DIV2K_valid_LR_bicubic/X2/'
config.valid.lrx4_folder_path = 'data2017/DIV2K_valid_LR_bicubic/X4/'
config.valid.lrx8_folder_path = 'data2017/DIV2K_valid_LR_bicubic/X8/'

config.train = edict()
config.train.hr_folder_path = 'data2017/DIV2K_train_HR/'
config.train.lrx2_folder_path = 'data2017/DIV2K_train_LR_bicubic/X2/'
config.train.lrx4_folder_path = 'data2017/DIV2K_train_LR_bicubic/X4/'
config.train.lrx8_folder_path = 'data2017/DIV2K_train_LR_bicubic/X8/'

config.train.batch_size = 4  # use large number if you have enough memory
config.train.in_patch_size = 64
config.train.out_patch_size = config.model.scale * config.train.in_patch_size
config.train.batch_size_each_folder = 30
config.train.log_write = False

## Adam
config.train.lr_init = 1e-4 # 6e-6
config.train.lr_decay = 0.90
config.train.decay_iter = 10
config.train.beta1 = 0.90

## initialize G
config.train.n_epoch = 300

## for run models generating images
config.TEST = edict()
config.TEST.mode = 'reflect'
config.TEST.type = 'odd'
config.TEST.padnumber = 1


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
