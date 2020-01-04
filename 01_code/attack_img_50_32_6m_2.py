#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
#import cv2
import scipy.stats as st
from timeit import default_timer as timer
import time

import tensorflow as tf
from nets import inception_v3, inception_resnet_v2, resnet_v2,inception_v4

slim = tf.contrib.slim

# 解决tf.app.flags 元素被重复定义问题
run_flag = False

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
        
if run_flag :
    del_all_flags(FLAGS)
    

model_path = "~/00_project/02_model_renamed"

# 定义flag
tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v3', model_path + '/inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_adv_inception_v3',  model_path + '/adv_inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens4_adv_inception_v3',  model_path + '/ens4_adv_inception_v3.ckpt', 'Path to checkpoint for inception network.')


tf.flags.DEFINE_string(
    'checkpoint_path_ens_adv_inception_resnet_v2',  model_path + '/ens_adv_inception_resnet_v2.ckpt', 'Path to checkpoint for inception network.')


tf.flags.DEFINE_string(
    'checkpoint_path_ens3_adv_inception_v3',  model_path + '/ens3_adv_inception_v3.ckpt', 'Path to checkpoint for inception network.')

# tf.flags.DEFINE_string(
#     'checkpoint_path_resnet_v2_101',  model_path + '/resnet_v2_101.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v4',  model_path + '/inception_v4.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_float(
    'prob', 0.5, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer(
    'image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'sig', 4, 'gradient smoothing')

tf.flags.DEFINE_integer(
    'kernlen', 7, 'kernlen')

tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_integer(
  'iterations', 50, 'iterations')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'orig_dir', '../images/', 'original directory with images.')

tf.flags.DEFINE_string(
    'input_target_dir', '../', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 32.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
        'batch_size', 5, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'augment_stddev', 0.005, 'stddev of image_augmentation random noise.')

tf.flags.DEFINE_float(
    'rotate_stddev', 0.005, 'stddev of image_rotation random noise.')

tf.flags.DEFINE_float(
    'alpha', 0.005, 'learning rate')

FLAGS = tf.flags.FLAGS

run_flag = True


def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  interval = (2*nsig+1.)/(kernlen)
  x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
  kern1d = np.diff(st.norm.cdf(x))
  kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
  kernel = kernel_raw/kernel_raw.sum()
  return kernel


# while_loop中的loop_vars：[x_input, x_orig, target_class_input, i, x_max, x_min, grad]
def graph(x, x_orig, target_class_input, i, x_max, x_min, grad):
  # eps = 2.0 * FLAGS.max_epsilon / 255.0
  eps = 2.0 * 32 / 255.0
  alpha = eps / FLAGS.iterations
 
  FLAGS.alpha = alpha
  #alpha = 2.0 / 255.0 #refer DI-2-FGSM

  momentum = FLAGS.momentum
  num_classes = 1001

  x_div = input_diversity(x)
  #x_div = x

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_v3, end_points_v3 = inception_v3.inception_v3(
        x_div, num_classes=num_classes, is_training=False, scope='OrgInceptionV3')
    
  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
        x_div, num_classes=num_classes, is_training=False, scope='AdvInceptionV3')

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
        x_div, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
        x_div, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2') 

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
        x_div, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')
    
#       with slim.arg_scope(resnet_v2.resnet_arg_scope()):
#     logits_resnet_v2_101, end_points_resnet_v2_101 = resnet_v2.resnet_v2_101(
#         x_div, num_classes=num_classes, is_training=False, scope='resnet_v2_101')

  with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
    logits_v4, end_points_v4 = inception_v4.inception_v4(
        x_div, num_classes=num_classes, is_training=False, scope='InceptionV4')
    
  one_hot_target_class = tf.one_hot(target_class_input, num_classes)

  logits = (logits_v3 + logits_adv_v3 + logits_ens4_adv_v3 +
            logits_ensadv_res_v2 + logits_ens3_adv_v3 + logits_v4) / 6

  auxlogits = (end_points_v3['AuxLogits'] + end_points_adv_v3['AuxLogits'] + \
    end_points_ens4_adv_v3['AuxLogits']  + end_points_ens3_adv_v3['AuxLogits'] + \
    end_points_ensadv_res_v2['AuxLogits'] + end_points_v4['AuxLogits']) / 6

  
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                  logits,
                                                  label_smoothing=0.0,
                                                  weights=1.0)
  cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                   auxlogits,
                                                   label_smoothing=0.0,
                                                   weights=0.4)
  noise = tf.gradients(cross_entropy, x)[0]

  kernel = gkern(FLAGS.kernlen, FLAGS.sig).astype(np.float32)
  stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
  stack_kernel = np.expand_dims(stack_kernel, 3)
  
  noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')

  noise = noise / tf.reshape(tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1), 
    [FLAGS.batch_size, 1, 1, 1])
  noise = momentum * grad + noise
  noise = noise / tf.reshape(tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1), 
    [FLAGS.batch_size, 1, 1, 1])
  x = x - alpha * tf.clip_by_value(tf.round(noise), -2, 2)
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  
  return x, x_orig, target_class_input, i, x_max, x_min, noise


def stop(x, x_orig, target_class_input, i, x_max, x_min, grad):
  return tf.less(i, FLAGS.iterations)

def load_target_class(input_target_dir):
  """Loads target classes."""
  with tf.gfile.Open(os.path.join(input_target_dir, 'target_class.csv')) as f:
    return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}

def load_images(input_dir, orig_dir, batch_shape):
  """Read png images from input directory in batches.
  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  orig_images = np.zeros(batch_shape)
    
  files = os.listdir(input_dir)
  filenames=[]
  idx = 0
  batch_size = batch_shape[0]

  for file in files:
    input_f = os.path.join(input_dir, file)
    orig_f  = os.path.join(orig_dir, file) #original images

    if (os.path.exists(input_f) == False and os.path.exists(orig_f) == False):
        print(input_f, 'or', orig_f, 'not exist')
        continue
    image = imread(input_f, mode='RGB').astype(np.float) / 255.0
    orig_img = imread(orig_f, mode='RGB').astype(np.float) / 255.0
    #image = cv2.imread(f).astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    orig_images[idx, :, :, :] = orig_img * 2.0 - 1.0
    
    filenames.append(file)
    idx += 1

    if idx == batch_size:
      yield filenames, images, orig_images
      filenames = []
      images = np.zeros(batch_shape)
      orig_images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images, orig_images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')
      #cv2.imwrite(f, (images[i, :, :, :] + 1.0) * 0.5)

from tensorflow.contrib.image import transform as images_transform
from tensorflow.contrib.image import rotate as images_rotate

def image_augmentation(x):
  # img, noise
  one = tf.fill([tf.shape(x)[0], 1], 1.)
  zero = tf.fill([tf.shape(x)[0], 1], 0.)
  transforms = tf.concat([one, zero, zero, zero, one, zero, zero, zero], axis=1)
  rands = tf.concat([tf.truncated_normal([tf.shape(x)[0], 6], stddev=FLAGS.augment_stddev), zero, zero], axis=1)
  return images_transform(x, transforms + rands, interpolation='BILINEAR')


def image_rotation(x):
  """ imgs, scale, scale is in radians """
  rands = tf.truncated_normal([tf.shape(x)[0]], stddev=FLAGS.rotate_stddev)
  return images_rotate(x, rands, interpolation='BILINEAR')
    
def input_diversity(input_tensor):
  """
  kernel_size=10
  p_dropout=0.1
  kernel = tf.divide(tf.ones((kernel_size,kernel_size,3,3),tf.float32),tf.cast(kernel_size**2,tf.float32))
  input_shape = input_tensor.get_shape()
  rand = tf.where(tf.random_uniform(input_shape) < tf.constant(p_dropout, shape=input_shape), 
    tf.constant(1., shape=input_shape), tf.constant(0., shape=input_shape))
  image_d = tf.multiply(input_tensor,rand)
  image_s = tf.nn.conv2d(input_tensor,kernel,[1,1,1,1],'SAME')
  input_tensor = tf.add(image_d,tf.multiply(image_s,tf.subtract(tf.cast(1,tf.float32),rand)))
  """
  input_tensor = image_augmentation(input_tensor)
  input_tensor = image_rotation(input_tensor)
  
  import time
  rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32, seed=time.time())
  rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  h_rem = FLAGS.image_resize - rnd
  w_rem = FLAGS.image_resize - rnd
  pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32, seed=time.time())
  pad_bottom = h_rem - pad_top
  pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32, seed=time.time())
  pad_right = w_rem - pad_left
  padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
  padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
  ret = tf.cond(tf.random_uniform(shape=[1], seed=time.time())[0] <= tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
  ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width], 
    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return ret


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  full_start = timer()
  eps = 2.0 * FLAGS.max_epsilon / 255.0

  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

  all_images_target_class = load_target_class(FLAGS.input_target_dir)
  print(type(all_images_target_class))

  with tf.Graph().as_default():            
    # Prepare graph
   
    x_orig  = tf.placeholder(tf.float32, shape=batch_shape)
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_orig + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_orig - eps, -1.0, 1.0)

    target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    i = tf.constant(0)
    grad = tf.zeros(shape=batch_shape) 
    
    x_adv, _, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, x_orig, target_class_input, i, x_max, x_min, grad])
    s1 = tf.train.Saver(slim.get_model_variables(scope='OrgInceptionV3'))
    s2 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
    s3 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
    s4 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
    s5 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
    s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))

    print('Created Graph')

    # Run computation
    with tf.Session() as sess:
      processed = 0.0
      s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
      s2.restore(sess, FLAGS.checkpoint_path_adv_inception_v3)
      s3.restore(sess, FLAGS.checkpoint_path_ens4_adv_inception_v3)
      s4.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)
      s5.restore(sess, FLAGS.checkpoint_path_ens3_adv_inception_v3)
      s6.restore(sess, FLAGS.checkpoint_path_inception_v4)
    
      print('Initialized Models... ')
                        
      load_ret = load_images(FLAGS.input_dir, FLAGS.orig_dir, batch_shape)
      index = 1
      for filenames, images, orig_images in load_ret:
        batch_start = timer()
        #print("log: images.shape = ", images.shape)
        #print("log: orig_images.shape = ", orig_images.shape)

        target_class = []
        for filename in filenames:
            # image_id = os.path.splitext(filename)[0]
            image_id = filename
            target_class.append(all_images_target_class[image_id])
            
        target_class_for_batch = (target_class + [0] * (FLAGS.batch_size - len(filenames)))
        adv_images = sess.run(x_adv, feed_dict={x_input: images, \
                                                x_orig: orig_images, target_class_input: target_class_for_batch})
        
        
        adv_images = (np.floor(np.abs(adv_images - images)*255)/255.0) * np.sign(adv_images - images) + images
        
        save_images(adv_images, filenames, FLAGS.output_dir)
        processed += FLAGS.batch_size
        batch_end = timer()
        print(index, processed,'use {} sec.'.format(batch_end - batch_start))
        index += 1 
      full_end = timer()
      print("DONE: Processed {} images in {} sec".format(processed, full_end - full_start))
      print('batch_size:{},iterations:{}, had save {} images to {}'.\
            format(FLAGS.batch_size,FLAGS.iterations,processed, FLAGS.output_dir))
    
      # write log
      log_file = open('logging.txt','a')
      time_now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
        
      log_file.write('\n{} : {}\n'.format(time_now,FLAGS.output_dir)) 
      log_file.write("DONE: Processed {} images in {} sec\n".format(processed, full_end - full_start))
      log_file.write('\taugment_stddev:{},sig:{},rotate_stddev:{},momentum:{},alpha:{},max_epsilon:{},kernlen:{}\n'.\
                format(FLAGS.augment_stddev,FLAGS.sig,FLAGS.rotate_stddev,\
                     FLAGS.momentum,FLAGS.alpha,FLAGS.max_epsilon,FLAGS.kernlen))

      log_file.write('\tbatch_size:{},iterations:{},image_resize:{},prob:{},, had save {} images to {}\n'.\
            format(FLAGS.batch_size,FLAGS.iterations,FLAGS.image_resize,FLAGS.prob,processed, FLAGS.output_dir))

      log_file.close()
        
      
if __name__ == '__main__':
  #from tensorflow.compat.v1 import ConfigProto
  #from tensorflow.compat.v1 import InteractiveSession

  #config = ConfigProto()
  #config.gpu_options.allow_growth = True 
  #session = InteractiveSession(config=config)
  # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

  tf.logging.set_verbosity(tf.logging.WARN)
  tf.app.run()
