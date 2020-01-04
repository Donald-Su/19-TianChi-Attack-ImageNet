#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import time

import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from check_pixs_attack import check_pixs_attack
# added end

import argparse #解析命令行

import csv
import os
import cv2 as cv
import PIL
import numpy as np

import tempfile
from urllib.request import urlretrieve
import tarfile

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def inception(image, reuse):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        #logits = logits[:,1:] # ignore background class
        probs = tf.nn.softmax(logits) # probabilities
    return logits, probs

def img_preprocessing(img_path):
    img = PIL.Image.open(img_path)            
    img = img.convert("RGB")
    big_dim = max(img.width, img.height)
    wide = img.width > img.height
    new_w = 299 if not wide else int(img.width * 299 / img.height)
    new_h = 299 if wide else int(img.height * 299 / img.width)
    img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
    img = (np.asarray(img) / 255.0).astype(np.float32)
    return img

def parse_args():
    parser = argparse.ArgumentParser(description="check acc")
    parser.add_argument('--ckpt_file', required=True,
                        help='model checkpoint file')
    parser.add_argument('--csv_file', default='dataset1.csv',
                        help='model checkpoint file')
    parser.add_argument('--scope_name', required=True,
                        help='different ckpt file have the different scope name')
    parser.add_argument('--img_folder', default=True,
                        help='img_folder')
    parser.add_argument('--cal_attack', required=True,
                        help='Is cal attack acc?')
    # parser.add_argument('--cal_recongnize', required=True,
    #                     help='Is cal recongnize acc?')

    args = parser.parse_args()
    return args.ckpt_file, args.scope_name,args.img_folder, args.cal_attack, args.csv_file


def cal_score(img_folder, ckpt_file,scope_name, csv_file, cal_attack = True):
    image = tf.Variable(tf.zeros((299, 299, 3)))
    logits, probs = inception(image, reuse=False)

    filenames = []
    TrueLabels = []
    TargetClasses = []
    
    print(csv_file)
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = os.path.join(row["ImageId"])
            filenames.append(filepath)
            TrueLabel = int(row["TrueLabel"])
            TrueLabels.append(TrueLabel)
            TargetClass = int(row["TargetClass"])
            TargetClasses.append(TargetClass)
    
    print("filenames = ", len(filenames))
    
    score = 0
    succ = 0
    failed = 0
    half_succ = 0

    # 计算像素值超过32的
    more32_succ = 0
    more32_half = 0
    more32_failed = 0
      
    # print(scope_name)
    restore_vars = [
        var for var in tf.global_variables()
        if var.name.startswith(str(scope_name))
    ]
    # print(restore_vars)
    saver = tf.train.Saver(restore_vars)
    saver.restore(sess, ckpt_file)

    cnt = 1
    orig_folder = 'images'  #原始图像
    for i in range(1216):
        image_path = os.path.join(img_folder, filenames[i])
        if os.path.exists(image_path) == False:
            print(image_path, 'not exist, total:',cnt)   
            cnt += 1
            continue  
            
        img = img_preprocessing(image_path)
        
        ret_img = np.array([])

        # 对攻击后的图片进行计算
        if cal_attack:
            #像素值是否超过32
            check_ret,ret_img = check_pixs_attack(orig_folder,img_folder, \
                                          os.path.splitext(filenames[i])[0], ret_img)
            
            p = sess.run(probs, feed_dict={image: ret_img/255.0})[0]
            
            topk = list(p.argsort()[-1:][::-1])     
            if topk[0] == TargetClasses[i]:                
                score += 1
                succ += 1
                if check_ret == False: # 超过32
                    more32_succ += 1
                
        
            elif(topk[0]!=TargetClasses[i] and topk[0]!=TrueLabels[i]):                  
                score += 0.5
                half_succ += 1
                if check_ret == False: # 超过32
                    more32_half += 1
                    
            else:
                score += 0
                failed += 1
                if check_ret == False: # 超过32
                    more32_failed += 1
        
        # 对原始图像进行评估
        else:
            p = sess.run(probs, feed_dict={image: img})[0]
            topk = list(p.argsort()[-1:][::-1])
            
            if topk[0] == TrueLabels[i]:
                score += 1
                succ += 1
            elif(topk[0]!=TargetClasses[i] and topk[0]!=TrueLabels[i]):
                half_succ += 1 #既不是truelabel，又不是target
            else:
                failed += 1
    if cal_attack:
        return succ, half_succ, failed, score, more32_succ, more32_half, more32_failed
    else:
        return succ, half_succ, failed, score 
    
def main():
    ckpt_file, scope_name, img_folder, cal_attack, csv_file = parse_args()
    print(cal_attack, type(cal_attack))

    if cal_attack == 'True':
        score = cal_score(img_folder, ckpt_file, scope_name, csv_file)        
        
        # succ, half_succ, failed, score
        print('\n',img_folder)
        print(ckpt_file,'attack:', score[0], score[1], score[2],':', score[3])
        print(ckpt_file,'pix:', score[4], score[5], score[6],':', \
                   (score[4]+score[5]+score[6]))
       

                   
        score_del = score[3]- score[4] - score[5]*0.5
        print('\tdelete more 32:', score[0]-score[4], score[1]-score[5], \
              score[2]-score[6],':', score_del)
        # 写日志文件
        log_file = open('score_attack.txt','a')
        time_now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
        
        log_file.write('\n{} :\n {}\n'.format(time_now, img_folder)) 
        log_file.write('\t{} {} {} {} {} {} {}\n'.format(ckpt_file,'attack:', score[0], score[1], score[2],':', score[3]))
        log_file.write('\t{} {} {} {} {} {} {}\n'.format(ckpt_file,'pix:', score[4], score[5], score[6],':', \
                   (score[4]+score[5]+score[6])))
        log_file.write('\t{} {} {} {} {} {}\n'.format('\tdelete more 32:', score[0]-score[4], score[1]-score[5], \
              score[2]-score[6],':', score_del))
        log_file.close()

    else:
        score = cal_score(img_folder, ckpt_file,scope_name,csv_file, cal_attack=False)
        print('\n',img_folder)

        print(ckpt_file,'recognize:', score[0], score[1], score[2], ':', score[3])
        print('\n')
                       
        time_now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())  
                       
        log_file_recong = open('score_recognize.txt','a')
        log_file_recong.write('\n{} : \n{}\n'.format(time_now, img_folder))
        log_file_recong.write('\t{} recognize: {} {} {} {} {}\n'.format(ckpt_file, score[0], score[1], score[2], ':', score[3]))
        log_file_recong.close()
                   
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.WARN)
    main()
