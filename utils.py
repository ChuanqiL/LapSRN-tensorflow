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
def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5],
                      [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.float32(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    return np.uint8(rgb.dot(xform.T))


def psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def readAndSnap(hr_directory, test_hr_img_file, zoom):
    ''' Highly modularized read, and snap
    '''
    ###=================== read image from filename test_hr_img_file ================###
    test_hr_img = scipy.misc.imread(
        hr_directory + test_hr_img_file, mode='RGB').astype(np.float32)

    ###== Snap the image to a shape that's compatible with the generator (2x, 4x) ==###
    s = zoom * 2
    by, bx = test_hr_img.shape[0] % s, test_hr_img.shape[1] % s
    test_hr_groundtruth = test_hr_img[
        by - by // 2:test_hr_img.shape[0] - by // 2, bx - bx // 2:
        test_hr_img.shape[1] - bx // 2, :]

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
                                                 -crop_num, :]
    print("[*] save images")
    scipy.misc.imsave(save_dir + test_hr_gen_file,
                      test_hr_gen_output_crop.astype(np.uint8))

    ###======================= Calculate Metrics =============================###
    psnr_bicubic = psnr(
        rgb2ycbcr(test_hr_groundtruth)[:, :, 0],
        rgb2ycbcr(test_hr_gen_output_crop)[:, :, 0])
    ssim_bicubic = ssim(
        rgb2ycbcr(test_hr_groundtruth)[:, :, 0],
        rgb2ycbcr(test_hr_gen_output_crop)[:, :, 0])
    f.write(test_hr_img_file + '\t')
    f.write(format(psnr_bicubic, '.4f') + ',')
    f.write(format(ssim_bicubic, '.4f') + '\n')
    return
