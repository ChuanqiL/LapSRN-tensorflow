#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, random
import numpy as np
import scipy

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import *

###====================== HYPER-PARAMETERS ===========================###
batch_size = config.train.batch_size
patch_size = config.train.in_patch_size
ni = int(np.sqrt(config.train.batch_size))


def compute_charbonnier_loss(tensor1, tensor2, is_mean=True):
    epsilon = 1e-6
    if is_mean:
        loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))
    else:
        loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))

    return loss


def load_file_list():
    train_hr_file_list = []
    train_lr_file_list = []
    valid_hr_file_list = []
    valid_lr_file_list = []

    directory = config.train.hr_folder_path
    for filename in [
            y for y in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, y))
    ]:
        train_hr_file_list.append("%s%s" % (directory, filename))

    directory = config.train.lr_folder_path
    for filename in [
            y for y in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, y))
    ]:
        train_lr_file_list.append("%s%s" % (directory, filename))

    directory = config.valid.hr_folder_path
    for filename in [
            y for y in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, y))
    ]:
        valid_hr_file_list.append("%s%s" % (directory, filename))

    directory = config.valid.lr_folder_path
    for filename in [
            y for y in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, y))
    ]:
        valid_lr_file_list.append("%s%s" % (directory, filename))

    return sorted(train_hr_file_list), sorted(train_lr_file_list), sorted(valid_hr_file_list), sorted(valid_lr_file_list)


def prepare_nn_data(hr_img_list, lr_img_list, idx_img=None):
    i = np.random.randint(len(hr_img_list)) if (idx_img is None) else idx_img

    input_image = get_imgs_fn(lr_img_list[i])
    output_image = get_imgs_fn(hr_img_list[i])
    scale = int(output_image.shape[0] / input_image.shape[0])
    assert scale == config.model.scale

    out_patch_size = patch_size * scale
    input_batch = np.empty([batch_size, patch_size, patch_size, 3])
    output_batch = np.empty([batch_size, out_patch_size, out_patch_size, 3])

    for idx in range(batch_size):
        in_row_ind = random.randint(0, input_image.shape[0] - patch_size)
        in_col_ind = random.randint(0, input_image.shape[1] - patch_size)

        input_cropped = augment_imgs_fn(
            input_image[in_row_ind:in_row_ind + patch_size, in_col_ind:
                        in_col_ind + patch_size])
        input_cropped = normalize_imgs_fn(input_cropped)
        input_cropped = np.expand_dims(input_cropped, axis=0)
        input_batch[idx] = input_cropped

        out_row_ind = in_row_ind * scale
        out_col_ind = in_col_ind * scale
        output_cropped = output_image[out_row_ind:out_row_ind + out_patch_size,
                                      out_col_ind:out_col_ind + out_patch_size]
        output_cropped = normalize_imgs_fn(output_cropped)
        output_cropped = np.expand_dims(output_cropped, axis=0)
        output_batch[idx] = output_cropped

    return input_batch, output_batch


def train(binary=False):
    save_dir = "%s/%s_train" % (config.model.result_path,
                                tl.global_flag['mode'])
    checkpoint_dir = "%s" % (config.model.checkpoint_path)
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###========================== DEFINE MODEL ============================###
    t_image = tf.placeholder(
        'float32', [batch_size, patch_size, patch_size, 3],
        name='t_image_input')
    t_target_image = tf.placeholder(
        'float32', [
            batch_size, patch_size * config.model.scale,
            patch_size * config.model.scale, 3
        ],
        name='t_target_image')
    t_target_image_down = tf.image.resize_images(
        t_target_image,
        size=[patch_size * 2, patch_size * 2],
        method=0,
        align_corners=False)

    net_image2, net_grad2, net_image1, net_grad1 = LapSRN(
        t_image, reuse=False, is_train=True, binary=binary)
    net_image2.print_params(False)

    ## test inference
    net_image2_test, net_grad2_test, net_image1_test, _ = LapSRN(
        t_image, reuse=True, is_train=False, binary=binary)

    ###========================== DEFINE TRAIN OPS ==========================###
    mse_loss2 = compute_charbonnier_loss(net_image2.outputs, t_target_image, is_mean=True)
    mse_loss1 = compute_charbonnier_loss(net_image1.outputs, t_target_image_down, is_mean=True)
    mse_loss = mse_loss1 + mse_loss2 * 4
    
    mse_loss2_test = compute_charbonnier_loss(net_image2_test.outputs, t_target_image, is_mean=True)
    mse_loss1_test = compute_charbonnier_loss(net_image1_test.outputs, t_target_image_down, is_mean=True)
    mse_loss_test = mse_loss1_test + mse_loss2_test * 4
    
    g_vars = get_variables_with_name_in_binary_training('LapSRN', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(config.train.lr_init, trainable=False)

    # g_optim = tf.train.AdamOptimizer(lr_v, beta1=config.train.beta1).minimize(mse_loss, var_list=g_vars)
    g_loss_gvs = tf.train.AdamOptimizer(
        lr_v, beta1=config.train.beta1).compute_gradients(
            mse_loss, var_list=g_vars)
    clipped_g_loss_gvs = [(tf.clip_by_value(grad, -5.0, 5.0)
                           if grad is not None else None, var)
                          for grad, var in g_loss_gvs]
    g_loss_grads_and_vars = process_grads(clipped_g_loss_gvs)
    g_optim = tf.train.AdamOptimizer(
        lr_v, beta1=config.train.beta1).apply_gradients(g_loss_grads_and_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    if binary:
        tl.files.load_and_assign_npz(
            sess=sess,
            name=checkpoint_dir + '/params_lapsrn_b.npz',
            network=net_image2)
        tl.files.load_and_assign_npz(
            sess=sess,
            name=checkpoint_dir + '/params_lapsrn_b.npz',
            network=net_image2_test)
    else:
        tl.files.load_and_assign_npz(
            sess=sess,
            name=checkpoint_dir + '/params_lapsrn.npz',
            network=net_image2)
        tl.files.load_and_assign_npz(
            sess=sess,
            name=checkpoint_dir + '/params_lapsrn.npz',
            network=net_image2_test)

    ###========================== PRE-LOAD DATA ===========================###
    train_hr_list, train_lr_list, valid_hr_list, valid_lr_list = load_file_list(
    )

    ###========================== Intermediate validation ===============================###
    sample_ind = 53
    sample_input_imgs, sample_output_imgs = prepare_nn_data(
        valid_hr_list, valid_lr_list, sample_ind)
    tl.vis.save_images(
        truncate_imgs_fn(sample_input_imgs), [ni, ni],
        save_dir + '/train_sample_input.png')
    tl.vis.save_images(
        truncate_imgs_fn(sample_output_imgs), [ni, ni],
        save_dir + '/train_sample_output.png')

    ###========================== Training ====================###
    sess.run(tf.assign(lr_v, config.train.lr_init))
    print(" ** learning rate: %f" % config.train.lr_init)

    for epoch in range(config.train.n_epoch):
        ## update learning rate
        if epoch != 0 and (epoch % config.train.decay_iter == 0):
            lr_decay = config.train.lr_decay**(
                epoch // config.train.decay_iter)
            lr = config.train.lr_init * lr_decay
            sess.run(tf.assign(lr_v, lr))
            print(" ** learning rate: %f" % (lr))

        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0
        total_mse_loss_test, n_test_iter = 0, 0

        ## load image data
        idx_list = np.random.permutation(len(train_hr_list))
        for idx_file in range(len(idx_list)):
            batch_input_imgs, batch_output_imgs = prepare_nn_data(
                train_hr_list, train_lr_list, idx_file)
            errM, _ = sess.run([mse_loss, g_optim], {
                t_image: batch_input_imgs,
                t_target_image: batch_output_imgs
            })
            total_mse_loss += errM
            n_iter += 1

        ## loss on valid data
        test_idx_list = np.random.permutation(len(valid_hr_list))
        for test_idx_file in range(len(test_idx_list)):
            batch_input_imgs, batch_output_imgs = prepare_nn_data(
                valid_hr_list, valid_lr_list, test_idx_file)
            errTest = sess.run(mse_loss_test, {
                t_image: batch_input_imgs,
                t_target_image: batch_output_imgs
            })
            total_mse_loss_test += errTest
            n_test_iter += 1

        print("[*] Epoch: [%2d/%2d] time: %4.4fs, train mse: %.6f, valid mse: %.6f" %
              (epoch, config.train.n_epoch, time.time() - epoch_time,
               total_mse_loss / n_iter, total_mse_loss_test / n_test_iter))

        ## save model and evaluation on sample set
        if (epoch != 0) and (epoch % 1 == 0):
            if binary:
                tl.files.save_npz(
                    net_image2.all_params,
                    name=checkpoint_dir +
                    '/params_lapsrn_b.npz',
                    sess=sess)
            else:
                tl.files.save_npz(
                    net_image2.all_params,
                    name=checkpoint_dir +
                    '/params_lapsrn.npz',
                    sess=sess)
            sample_out, sample_grad_out = sess.run(
                [net_image2_test.outputs, net_grad2_test.outputs], {
                    t_image: sample_input_imgs
                })  #; print('gen sub-image:', out.shape, out.min(), out.max())
            tl.vis.save_images(
                truncate_imgs_fn(sample_out), [ni, ni],
                save_dir + '/train_predict_%d.png' % epoch)
            tl.vis.save_images(
                truncate_imgs_fn(np.abs(sample_grad_out)), [ni, ni],
                save_dir + '/train_grad_predict_%d.png' % epoch)


def test(read_directory, binary=False, zoom=4):
    ###====================== PRE-LOAD DATA ===========================###
    # Read filename list from file_directory+'HR/'
    hr_directory = read_directory + 'HR/'
    lr_directory = read_directory + 'LRx{}/'.format(zoom)
    test_hr_img_list = sorted(
        tl.files.load_file_list(
            path=hr_directory, regx='.*.jpg', printable=False))
    test_hr_img_list.extend(
        sorted(
            tl.files.load_file_list(
                path=hr_directory, regx='.*.png', printable=False)))

    ## create folders to save result images
    model_label = "lapsrnx{}".format(zoom)
    if binary:
        model_label += '_b'
    save_dir = read_directory + model_label + '/'
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = config.model.checkpoint_path

    f = open(read_directory + 'metrics_' + model_label + '.txt', 'w+')

    run_times = 0
    for test_hr_img_file in test_hr_img_list:
        test_hr_groundtruth, test_lr_img_file = readAndSnap(
            hr_directory, test_hr_img_file, zoom)

        test_lr_img = scipy.misc.imread(
            lr_directory + test_lr_img_file, mode='RGB').astype(np.float32)

        ###========================== Padding the LR image =============================###
        test_lr_img_input = np.lib.pad(
            test_lr_img, ((config.TEST.padnumber, config.TEST.padnumber),
                          (config.TEST.padnumber, config.TEST.padnumber), (0,
                                                                           0)),
            config.TEST.mode,
            reflect_type=config.TEST.type)

        ###========================== DEFINE MODEL ============================###
        test_lr_img_input = (
            test_lr_img_input / 127.5) - 1  # rescale to ［－1, 1]
        size = test_lr_img_input.shape
        print('Input size: %s,%s,%s' % (size[0], size[1], size[2]))
        t_image = tf.placeholder(
            'float32', [None, size[0], size[1], size[2]], name='input_image')
        
        if run_times == 0:
            if zoom==2:
                _, _, net_g, _ = LapSRN(t_image, is_train=False, reuse=False, binary=binary)
            else:
                net_g, _, _, _ = LapSRN(t_image, is_train=False, reuse=False, binary=binary)

            ###========================== RESTORE G =============================###
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False))
            tl.layers.initialize_global_variables(sess)
            if binary:
                load_and_assign_npz_with_binary(
                    sess=sess,
                    name=checkpoint_dir + '/params_lapsrn_b.npz',
                    network=net_g,
                    binary=binary)
            else:
                tl.files.load_and_assign_npz(
                    sess=sess,
                    name=checkpoint_dir + '/params_lapsrn.npz',
                    network=net_g)
        else:
            if zoom==2:
                _, _, net_g, _ = LapSRN(t_image, is_train=False, reuse=True, binary=binary)
            else:
                net_g, _, _, _ = LapSRN(t_image, is_train=False, reuse=True, binary=binary)

        ###================ EVALUATE on Model and generate image ================###
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [test_lr_img_input]})
        print("took: %4.4fs" % (time.time() - start_time))
        print("LR size: %s /  generated HR size: %s" %
              (size, out.shape
               ))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        # test_hr_gen_output = rescale_imgs_fn(out[0])
        test_hr_gen_output = truncate_imgs_fn(out[0])
        test_hr_gen_output = (test_hr_gen_output + 1) * 127.5

        cropSaveCalculate(test_hr_groundtruth, test_hr_gen_output, save_dir,
                          test_hr_img_file, model_label, zoom, f)
        run_times += 1

    f.close()
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--mode',
        choices=['train', 'test'],
        default='train',
        help='select mode')
    parser.add_argument(
        '--test_dir',
        type=str,
        default='BSD100/',
        help='BSD100/, Set5/, Set14/')
    parser.add_argument(
        '--binary', 
        type=bool, 
        default=False, 
        help='enable binary network')
    parser.add_argument(
        '--zoom', 
        type=int, 
        default=4, 
        help='super-resolution scale')
    
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    if tl.global_flag['mode'] == 'train':
        train(binary=args.binary)
    elif tl.global_flag['mode'] == 'test':
        test('../srgan_binary/'+args.test_dir, binary=args.binary, zoom=args.zoom)
    else:
        raise Exception("Unknow --mode")
