import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *

import scipy
import math
import numpy as np

from numpy.lib.stride_tricks import as_strided as ast
import scipy.ndimage.filters as Filter
import scipy.misc
from skimage.measure import compare_ssim as ssim
from config import config, log_config


def get_imgs_fn(file_name):
    return scipy.misc.imread(file_name, mode='RGB')


def augment_imgs_fn(x, add_noise=True):
    return x + 0.1 * x.std() * np.random.random(x.shape)


def normalize_imgs_fn(x):
    x = x * (2. / 255.) - 1.
    # x = x * (1./255.)
    return x


def rescale_imgs_fn(x):
	x = (2 * x - (np.amax(x) + np.amin(x))) / (np.amax(x) - np.amin(x))
	return x


def truncate_imgs_fn(x):
	''' deprecate '''
	x = np.where(x >= -1., x, -1)
	x = np.where(x <= 1., x, 1)
	return x

#--------------------------------------------------------------------------------

def readAndSnap(hr_directory, test_hr_img_file, zoom):
    ''' Highly modularized read, and snap
    '''
    ###=================== read image from filename test_hr_img_file ================###
    test_hr_img = scipy.misc.imread(
        hr_directory + test_hr_img_file, mode='RGB')

    ###== Snap the image to a shape that's compatible with the generator (2x, 4x) ==###
    by, bx = test_hr_img.shape[0] % zoom, test_hr_img.shape[1] % zoom
    test_hr_groundtruth = test_hr_img[:test_hr_img.shape[0] - by, :test_hr_img.shape[1] - bx, :]

    ###=================== read corresponding LR image ================###
    if 'png' in test_hr_img_file:
        test_lr_img_file = os.path.splitext(
            test_hr_img_file)[0] + 'x{}.png'.format(zoom)
    else:
        test_lr_img_file = os.path.splitext(
            test_hr_img_file)[0] + 'x{}.jpg'.format(zoom)

    return test_hr_groundtruth, test_lr_img_file


def cropSaveCalculate(test_hr_groundtruth, test_hr_gen_output, save_dir,
                      test_hr_img_file, model_label, zoom, f):
    ''' Highly modularized crop, save, and metrics calculation
    '''
    if 'png' in test_hr_img_file:
        test_hr_gen_file = os.path.splitext(
            test_hr_img_file)[0] + '_' + model_label + '.png'
    else:
        test_hr_gen_file = os.path.splitext(
            test_hr_img_file)[0] + '_' + model_label + '.jpg'

    ###======================= Crop and save generated =======================###
    crop_num = config.TEST.padnumber * zoom
    test_hr_gen_output_crop = test_hr_gen_output[crop_num:-crop_num, crop_num:
                                                 -crop_num, :].astype(np.uint8)
    print("[*] save images")
    scipy.misc.imsave(save_dir + test_hr_gen_file,
                      test_hr_gen_output_crop)

    return
