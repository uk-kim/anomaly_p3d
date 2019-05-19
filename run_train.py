import os
import sys
import argparse
import glob

import tensorflow as tf
import numpy as np

from config import *
import dbread as db
from AAE import AAE


''' parsing and configuration '''
def parse_args():
    desc="Implementation of AAE with P3D models using Tensorflow for Anomaly Detection in Video Scenes"
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--initial_learning_rate', type=float, default=INITIAL_LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_frames_per_clip', type=int, default=NUM_FRAMES_PER_CLIP)
    parser.add_argument('--dataset_shuffle', type=bool, default=True)

    parser.add_argument('--log_dir', type=str, default=LOG_DIR)
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR)
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR)
    
    parser.add_argument('--z_dim', type=int, default=LATENT_DIM)
    parser.add_argument('--image_crop_size', type=int, default=IMAGE_CROP_SIZE)
    
    return check_args(parser.parse_args())

''' checking arguments'''
def check_args(args):
    # --num_epochs
    try:
        assert args.num_epochs <= 0
    except:
        print("number of epochs must be larger than or equal to one")
    
    # --initial_learning_rate
    try:
        assert args.initial_learning_rate > 0
    except:
        print("initial_learning_rate must be positive")
    
    # --batch_size
    try:
        assert args.batch_size > 0
    except:
        print("batch size must be larger than or equal to one")
    
    # --num_frames_per_clip
    try:
        assert args.num_frames_per_clip > 0
    except:
        print("number of frames per clip must be larger than or equal to one. (8 is recommanded)")
    
    # --dataset_shuffle
    try:
        assert args.dataset_shuffle == True or args.dataset_shuffle == False
    except:
        print("dataset shuffle flag must be boolean type")
    
    # --log_dir
    try:
        os.mkdir(args.log_dir)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.log_dir + '/*')
    for f in files:
        os.remove(f)
    
    # --model_dir
    try:
        os.mkdir(args.model_dir)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.model_dir + '/*')
    for f in files:
        os.remove(f)
        
    # --log_dir
    try:
        os.mkdir(args.result_dir)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.result_dir + '/*')
    for f in files:
        os.remove(f)
        
    # --z_dim
    try:
        assert args.z_dim > 0
    except:
        print("z dimension(latent dimension) must be larger than or equal to one")
    
    # --image_crop_size
    try:
        assert args.image_crop_size > 0
    except:
        print("image cropping size must be larger than or equal to one. (224 is recommanded)")
    
    return args


""" Main Function """
def main(args):
    """ Parameters """
    batch_size            = args.batch_size
    num_frames_per_clip   = args.num_frames_per_clip
    dataset_shuffle       = args.dataset_shuffle
    num_epochs            = args.num_epochs
    initial_learning_rate = args.initial_learning_rate
    z_dim                 = args.z_dim
    image_crop_size       = args.image_crop_size
    
    ''' Dataset Reader'''
    reader=db.DBreader(batch_size=batch_size, n_frames_clip=num_frames_per_clip, 
                       resize=[image_crop_size, image_crop_size], shuffle=dataset_shuffle)

    ''' Build Graph'''
    # input placeholder
    x = tf.placeholder(tf.float32, 
                       shape=[batch_size, num_frames_per_clip, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 1])
    z_sample = tf.placeholder(tf.float32, 
                              shape=[batch_size, 1, image_crop_size//4, image_crop_size//4, z_dim])
    
    ''' Network Architecture'''
    model = AAE()

    y, z, neg_marginal_likelihood, D_loss, G_loss = model.adversarial_autoencoder(x, z_sample)
    
    ''' Optimization '''
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if "Discriminator" in var.name]
    g_vars = [var for var in t_vars if "Encoder" in var.name]
    ae_vars = [var for var in t_vars if "Encoder" or "Decoder" in var.name]

    train_op_ae = tf.train.AdamOptimizer(initial_learning_rate).minimize(neg_marginal_likelihood, var_list=ae_vars)
    train_op_d  = tf.train.AdamOptimizer(initial_learning_rate/5).minimize(D_loss, var_list=d_vars)
    train_op_g  = tf.train.AdamOptimizer(initial_learning_rate).minimize(G_loss, var_list=g_vars)
    
    ''' Training '''
    total_batch = reader.n_train_clips // batch_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Variable Initialized")

        for epoch in range(num_epochs):
            # Train Dataset Random Shuffling
            reader.initialize(True)

            for i in range(total_batch):
                train_x = reader.next_batch() / 255.

                # now here, generate z_sample by random noise
                # z_sample.shape.as_list()  --> sample's shape
                train_z_sample = np.random.random(z_sample.shape.as_list())

                # Reconstruction Loss
                _, loss_likelihood = sess.run([train_op_ae, neg_marginal_likelihood],
                                             feed_dict={x:train_x, z_sample:train_z_sample})

                # Discriminator loss
                _, d_loss = sess.run([train_op_d, D_loss],
                                    feed_dict={x:train_x, z_sample:train_z_sample})

                # Generator loss
                for _ in range(2):
                    _, g_loss = sess.run([train_op_g, G_loss],
                                        feed_dict={x:train_x, z_sample:train_z_sample})

                tot_loss = loss_likelihood + d_loss + g_loss
                print(" >> [%03d - %d/%d]: L_tot %03.2f, L_likelihood %03.2f, d_loss %03.2f, g_loss %03.2f" % (epoch, i, total_batch, tot_loss, loss_likelihood, d_loss, g_loss))

            # print cost every epoch
            print("epoch %03d: L_tot %03.2f, L_likelihood %03.2f, d_loss %03.2f, g_loss %03.2f" % (epoch, tot_loss, loss_likelihood, d_loss, g_loss))


if __name__ == '__main__':
    
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
       
    # main
    main(args)


def normalize(im):
    return im * (2.0 / 255.0) - 1


def denormalize(im):
    return (im + 1.) / 2.


def split_images(img, direction):
    tmp = np.split(img, 2, axis=2)
    img_A = tmp[0]
    img_B = tmp[1]
    if direction == 'AtoB':
        return img_A, img_B
    elif direction == 'BtoA':
        return img_B, img_A
    else:
        sys.exit("'--direction' should be 'AtoB' or 'BtoA'")


# Function for save the generated result
def save_visualization(X, nh_nw, save_path='./vis/sample.jpg'):
    nh, nw = nh_nw
    h, w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh, w * nw, 3))

    for n, x in enumerate(X):
        j = int(n / nw)
        i = int(n % nw)
        img[j * h:j * h + h, i * w:i * w + w, :] = x

    scipy.misc.imsave(save_path, img)


def main2():
    global_epoch = tf.Variable(0, trainable=False, name='global_step')
    global_epoch_increase = tf.assign(global_epoch, tf.add(global_epoch, 1))

    args = parser.parse_args()
    direction = args.direction
    filelist_train = args.train
    result_dir = args.out_dir + '/result'
    ckpt_dir = args.out_dir + '/checkpoint'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    total_epoch = args.epochs
    batch_size = args.batch_size

    database = db.DBreader(filelist_train, batch_size=batch_size, labeled=False, resize=[256, 512])

    sess = tf.Session()
    model = Pix2Pix(sess, batch_size)

    saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    total_batch = database.total_batch

    epoch = sess.run(global_epoch)
    while True:
        if epoch == total_epoch:
            break
        for step in range(total_batch):
            img_input, img_target = split_images(database.next_batch(), direction)
            img_target = normalize(img_target)
            img_input = normalize(img_input)

            loss_D = model.train_discrim(img_input, img_target)         # Train Discriminator and get the loss value
            loss_GAN, loss_L1 = model.train_gen(img_input, img_target)  # Train Generator and get the loss value

            if step % 100 == 0:
                print('Epoch: [', epoch, '/', total_epoch, '], ', 'Step: [', step, '/', total_batch, '], D_loss: ', loss_D, ', G_loss_GAN: ', loss_GAN, ', G_loss_L1: ', loss_L1)

            if step % 500 == 0:
                generated_samples = denormalize(model.sample_generator(img_input, batch_size=batch_size))
                img_target = denormalize(img_target)
                img_input = denormalize(img_input)

                img_for_vis = np.concatenate([img_input, generated_samples, img_target], axis=2)
                savepath = result_dir + '/output_' + 'EP' + str(epoch).zfill(3) + "_Batch" + str(step).zfill(6) + '.jpg'
                save_visualization(img_for_vis, (batch_size, 1), save_path=savepath)

        epoch = sess.run(global_epoch_increase)
        saver.save(sess, ckpt_dir + '/model_epoch'+str(epoch).zfill(3))
