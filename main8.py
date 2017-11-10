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
        loss = tf.reduce_mean(
            tf.reduce_mean(
                tf.sqrt(tf.square(tf.subtract(tensor1, tensor2)) + epsilon),
                [1, 2, 3]))
    else:
        loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.sqrt(tf.square(tf.subtract(tensor1, tensor2)) + epsilon),
                [1, 2, 3]))

    return loss


def load_file_list():
    train_hr_file_list = []
    train_lrx2_file_list = []
    train_lrx4_file_list = []
    train_lr_file_list = []
    valid_hr_file_list = []
    valid_lrx2_file_list = []
    valid_lrx4_file_list = []
    valid_lr_file_list = []
    
    directory = config.train.hr_folder_path
    for filename in [
            y for y in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, y))
    ]:
        train_hr_file_list.append("%s%s" % (directory, filename))

    directory = config.train.lrx2_folder_path
    for filename in [
            y for y in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, y))
    ]:
        train_lrx2_file_list.append("%s%s" % (directory, filename))

    directory = config.train.lrx4_folder_path
    for filename in [
            y for y in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, y))
    ]:
        train_lrx4_file_list.append("%s%s" % (directory, filename))

    directory = config.train.lrx8_folder_path
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

    directory = config.valid.lrx2_folder_path
    for filename in [
            y for y in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, y))
    ]:
        valid_lrx2_file_list.append("%s%s" % (directory, filename))

    directory = config.valid.lrx4_folder_path
    for filename in [
            y for y in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, y))
    ]:
        valid_lrx4_file_list.append("%s%s" % (directory, filename))

    directory = config.valid.lrx8_folder_path
    for filename in [
            y for y in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, y))
    ]:
        valid_lr_file_list.append("%s%s" % (directory, filename))

    return sorted(train_hr_file_list), sorted(train_lrx2_file_list), sorted(train_lrx4_file_list), \
        sorted(train_lr_file_list), sorted(valid_hr_file_list), sorted(valid_lrx2_file_list), \
        sorted(valid_lrx4_file_list), sorted(valid_lr_file_list)


def prepare_nn_data(hr_img_list, lrx2_img_list, lrx4_img_list, lr_img_list, idx_img=None):
    i = np.random.randint(len(hr_img_list)) if (idx_img is None) else idx_img

    input_image = get_imgs_fn(lr_img_list[i])
    middle1_image = get_imgs_fn(lrx4_img_list[i])
    middle2_image = get_imgs_fn(lrx2_img_list[i])
    output_image = get_imgs_fn(hr_img_list[i])
    scale = int(output_image.shape[0] / input_image.shape[0])
    assert scale == config.model.scale8

    mid1_patch_size = patch_size * 2
    mid2_patch_size = patch_size * 4
    out_patch_size = patch_size * scale
    input_batch = np.empty([batch_size, patch_size, patch_size, 3])
    middle1_batch = np.empty([batch_size, mid1_patch_size, mid1_patch_size, 3])
    middle2_batch = np.empty([batch_size, mid2_patch_size, mid2_patch_size, 3])
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

        mid1_row_ind = in_row_ind * 2
        mid1_col_ind = in_col_ind * 2
        middle1_cropped = output_image[mid1_row_ind:mid1_row_ind + mid1_patch_size,
                                      mid1_col_ind:mid1_col_ind + mid1_patch_size]
        middle1_cropped = normalize_imgs_fn(middle1_cropped)
        middle1_cropped = np.expand_dims(middle1_cropped, axis=0)
        middle1_batch[idx] = middle1_cropped

        mid2_row_ind = in_row_ind * 4
        mid2_col_ind = in_col_ind * 4
        middle2_cropped = output_image[mid2_row_ind:mid2_row_ind + mid2_patch_size,
                                      mid2_col_ind:mid2_col_ind + mid2_patch_size]
        middle2_cropped = normalize_imgs_fn(middle2_cropped)
        middle2_cropped = np.expand_dims(middle2_cropped, axis=0)
        middle2_batch[idx] = middle2_cropped

        out_row_ind = in_row_ind * scale
        out_col_ind = in_col_ind * scale
        output_cropped = output_image[out_row_ind:out_row_ind + out_patch_size,
                                      out_col_ind:out_col_ind + out_patch_size]
        output_cropped = normalize_imgs_fn(output_cropped)
        output_cropped = np.expand_dims(output_cropped, axis=0)
        output_batch[idx] = output_cropped

    return input_batch, middle1_batch, middle2_batch, output_batch


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
            patch_size * config.model.scale8, 3
        ],
        name='t_target_image')
    t_target_image_down = tf.placeholder(
        'float32', [
            batch_size, patch_size * config.model.scale,
            patch_size * config.model.scale8, 3
        ],
        name='t_target_image_down')
    t_target_image_down_more = tf.placeholder(
        'float32', [
            batch_size, patch_size * 2,
            patch_size * 2, 3
        ],
        name='t_target_image_down_more')

    net_image3, net_grad3, net_image2, net_grad2, net_image1, net_grad1 = LapSRN8(
        t_image, reuse=False, is_train=True, binary=binary)
    net_image3.print_params(False)

    ## test inference
    net_image3_test, net_grad3_test, net_image2_test, _, net_image1_test, _ = LapSRN8(
        t_image, reuse=True, is_train=False, binary=binary)

    with tf.variable_scope("LapSRN8", reuse=True):
        real_filters = tl.layers.get_variables_with_name('W_conv2d_real', True, False)
        binary_filters = tl.layers.get_variables_with_name('W_conv2d_binary', True, False)
        # Assign op
        assign_ops = []
        for i in range(len(real_filters)):
            alpha = tf.reduce_mean(tf.abs(real_filters[i]))
            assign_op = tf.assign(binary_filters[i], alpha * tf.sign(real_filters[i]))
            assign_ops.append(assign_op)

    ###========================== DEFINE TRAIN OPS ==========================###
    mse_loss3 = compute_charbonnier_loss(
        net_image3.outputs, t_target_image, is_mean=True)
    mse_loss2 = compute_charbonnier_loss(
        net_image2.outputs, t_target_image_down, is_mean=True)
    mse_loss1 = compute_charbonnier_loss(
        net_image1.outputs, t_target_image_down_more, is_mean=True)
    mse_loss = mse_loss1 + mse_loss2 * 4 + mse_loss3 * 16
    
    mse_loss3_test = compute_charbonnier_loss(
        net_image3_test.outputs, t_target_image, is_mean=True)
    mse_loss2_test = compute_charbonnier_loss(
        net_image2_test.outputs, t_target_image_down, is_mean=True)
    mse_loss1_test = compute_charbonnier_loss(
        net_image1_test.outputs, t_target_image_down_more, is_mean=True)
    mse_loss_test = mse_loss1_test + mse_loss2_test * 4 + mse_loss3_test * 16
    
    g_vars = get_variables_with_name_in_binary_training('LapSRN8', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(config.train.lr_init, trainable=False)

    # g_optim = tf.train.AdamOptimizer(lr_v, beta1=config.train.beta1).minimize(mse_loss, var_list=g_vars)
    g_loss_gvs = tf.train.AdamOptimizer(
        lr_v, beta1=config.train.beta1).compute_gradients(
            mse_loss, var_list=g_vars)
    clipped_g_loss_gvs = [(tf.clip_by_value(grad, -5.0, 5.0)
                           if grad is not None else None, var)
                          for grad, var in g_loss_gvs]
    g_loss_grads_and_vars = process_grads(clipped_g_loss_gvs, binary=True)
    g_optim = tf.train.AdamOptimizer(
        lr_v, beta1=config.train.beta1).apply_gradients(g_loss_grads_and_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    if binary:
        tl.files.load_and_assign_npz(
            sess=sess,
            name=checkpoint_dir + '/params_lapsrn8_b.npz',
            network=net_image3)
    else:
        tl.files.load_and_assign_npz(
            sess=sess,
            name=checkpoint_dir + '/params_lapsrn8.npz',
            network=net_image3)

    ###========================== PRE-LOAD DATA ===========================###
    train_hr_list, train_lrx2_list, train_lrx4_list, train_lr_list, valid_hr_list, valid_lrx2_list, valid_lrx4_list, valid_lr_list = load_file_list(
    )

    ###========================== Intermediate validation ===============================###
    sample_ind = 53
    sample_input_imgs, _, _, sample_output_imgs = prepare_nn_data(
        valid_hr_list, valid_lrx2_list, valid_lrx4_list, valid_lr_list, sample_ind)
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
            step_time = time.time()
            batch_input_imgs, batch_middle1_imgs, batch_middle2_imgs, batch_output_imgs = prepare_nn_data(
                train_hr_list, train_lrx2_list, train_lrx4_list, train_lr_list, idx_file)
            # Assign binary weight
            real_before, binary_before = sess.run([real_filters, binary_filters])
            for assign_op in assign_ops:
                sess.run(assign_op)

            errM, _ = sess.run([mse_loss, g_optim], {
                t_image: batch_input_imgs,
                t_target_image: batch_output_imgs,
                t_target_image_down: batch_middle2_imgs,
                t_target_image_down_more: batch_middle1_imgs
            })
            total_mse_loss += errM
            n_iter += 1

        ## loss on valid data
        test_idx_list = np.random.permutation(len(valid_hr_list))
        for test_idx_file in range(len(test_idx_list)):
            batch_input_imgs, batch_middle1_imgs, batch_middle2_imgs, batch_output_imgs = prepare_nn_data(
                valid_hr_list, valid_lrx2_list, valid_lrx4_list, valid_lr_list, test_idx_file)
            errTest = sess.run(mse_loss_test, {
                t_image: batch_input_imgs,
                t_target_image: batch_output_imgs,
                t_target_image_down: batch_middle2_imgs,
                t_target_image_down_more: batch_middle1_imgs
            })
            total_mse_loss_test += errTest
            n_test_iter += 1

        print("[*] Epoch: [%2d/%2d] time: %4.4fs, train mse: %.6f, valid mse: %.6f" %
              (epoch, config.train.n_epoch, time.time() - epoch_time,
               total_mse_loss / n_iter, total_mse_loss_test / n_test_iter))

        if binary:
            hs = open(checkpoint_dir + '/params_lapsrn8_b.txt'.format(zoom),"a+")
        else:
            hs = open(checkpoint_dir + '/params_lapsrn8.txt'.format(zoom),"a+")
        hs.write("{}, ".format(total_mse_loss / n_iter, '.6f'))
        hs.write("{}\n".format(total_mse_loss_test / n_test_iter, '.6f'))
        hs.close()
        
        ## save model and evaluation on sample set
        if (epoch != 0) and (epoch % 1 == 0):
            if binary:
                tl.files.save_npz(
                    net_image3.all_params,
                    name=checkpoint_dir + '/params_lapsrn8_b.npz',
                    sess=sess)
            else:
                tl.files.save_npz(
                    net_image3.all_params,
                    name=checkpoint_dir + '/params_lapsrn8.npz',
                    sess=sess)
            sample_out, sample_grad_out = sess.run(
                [net_image_test.outputs, net_grad_test.outputs], {
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
    model_label = "lapsrn8x{}".format(zoom)
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
            if zoom == 2:
                _, _, _, _, net_g, _ = LapSRN8(
                    t_image, is_train=False, reuse=False, binary=binary)
            elif zoom == 4:
                _, _, net_g, _, _, _ = LapSRN8(
                    t_image, is_train=False, reuse=False, binary=binary)
            else:
                net_g, _, _, _, _, _ = LapSRN8(
                    t_image, is_train=False, reuse=False, binary=binary)

            ###========================== RESTORE G =============================###
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False))
            tl.layers.initialize_global_variables(sess)
            if binary:
                load_and_assign_npz_with_binary(
                    sess=sess,
                    name=checkpoint_dir + '/params_lapsrn8_b.npz',
                    network=net_g,
                    binary=binary)
            else:
                tl.files.load_and_assign_npz(
                    sess=sess,
                    name=checkpoint_dir + '/params_lapsrn8.npz',
                    network=net_g)
        else:
            if zoom == 2:
                _, _, _, _, net_g, _ = LapSRN8(
                    t_image, is_train=False, reuse=True, binary=binary)
            elif zoom == 4:
                _, _, net_g, _, _, _ = LapSRN8(
                    t_image, is_train=False, reuse=True, binary=binary)
            else:
                net_g, _, _, _, _, _ = LapSRN8(
                    t_image, is_train=False, reuse=True, binary=binary)

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
        help='BSD100/, Set5/, Set14/, Urban100/, Manga109/')
    parser.add_argument(
        '--binary', type=bool, default=False, help='enable binary network')
    parser.add_argument(
        '--zoom', type=int, default=8, help='super-resolution scale')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    if tl.global_flag['mode'] == 'train':
        train(binary=args.binary)
    elif tl.global_flag['mode'] == 'test':
        test(
            '../srgan_binary/' + args.test_dir,
            binary=args.binary,
            zoom=args.zoom)
    else:
        raise Exception("Unknow --mode")
