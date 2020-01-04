#!/usr/bin/env python
# coding: utf-8

# ## 计算像素差值

import csv
import os
import cv2
import numpy as np
import tensorflow as tf

# 如果像素值少于32，True
# 像素值大于32，返回False
def check_pixs_attack(orig_folder,attack_folder,image_id, ret_img):
    if os.path.splitext(image_id)[1] == '.png':
        image_id = os.path.splitext(image_id)[0]
    orig_file = os.path.join(orig_folder, image_id + '.png')
    if os.path.exists(orig_file) == False:
        print(orig_file, 'not exitst')
        return False,ret_img

    attack_file = os.path.join(attack_folder, image_id +'.png')    
    if os.path.exists(attack_file) == False:
        print(attack_file, 'not exitst')
        return False,ret_img

    
    orig_img = cv2.imread(orig_file).astype(np.float)
    attack_img = cv2.imread(attack_file).astype(np.float)        
 
    x_max = np.clip(orig_img + 32, 0, 255)
    x_min = np.clip(orig_img - 32, 0, 255)
    
    ret_img = np.clip(attack_img, x_min, x_max)

    # print("【log】orig:{}, ret:{},ret_img.shape:{},width:{}, heigh:{}".format(type(orig_img),type(ret_img), \
    #                                                                        ret_img.shape, ret_img.width, ret_img.height))
    # print("【log】orig:{}, ret:{},ret_img.shape:{}".format(type(orig_img),type(ret_img), ret_img.shape))
    
    temp = cv2.absdiff(orig_img, attack_img)
    
    #print(attack_file, '1.max value:', np.max(temp))

    if(np.max(temp) > 32):
        print(attack_file, '2.super 32, max value:', np.max(temp))
        return False,ret_img
    else:    
        # print(attack_file, '<= 32, max value:', np.max(temp))
        return True,ret_img
        





