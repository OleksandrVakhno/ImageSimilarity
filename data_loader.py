#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:54:07 2019

@author: alex
"""

import sys
import numpy as np
from imageio import imread
from scipy import misc
import pickle
import os
import matplotlib.pyplot as plt
import cv2

"""Script to preprocess the omniglot dataset and pickle it into an array that's easy
    to index my character type"""

data_path = os.path.join('data/')
train_folder = os.path.join(data_path,'images_train')
valpath = os.path.join(data_path,'images_evaluation')

save_path = 'data/'





def loadimgs(path,n=0):
    #if data not already unzipped, unzip it.
    if not os.path.exists(path):
        print("unzipping")
        os.chdir(data_path)
        os.system("unzip {}".format(path+".zip" ))
    X=[]
    y = []
    cat_dict = {}
    curr_y = n
    #we load every alphabet seperately so we can isolate them later
    for category in os.listdir(path):
        if category==".DS_Store":
            continue
        print("loading picture: " + category)
        cat_dict[category] = [curr_y,None]
        category_path = os.path.join(path,category)
        #every letter/category has it's own column in the array, so  load seperately
        category_images=[]
        for i, filename in enumerate(os.listdir(category_path)):
            if i>25:
                break
            image_path = os.path.join(category_path, filename)
            image = cv2.imread(image_path)
            image_resized = cv2.resize(image, (256, 256))
            category_images.append(image_resized)
            y.append(curr_y)
            curr_y += 1
            cat_dict[category][1] = curr_y - 1
        try:
            print("=================================================================")
            X.append(np.stack(category_images))
        except ValueError as e:
            #print(e)
            print("error - category_images:", category_images)
    y = np.vstack(y)
    X = np.stack(X)
    return X,y,cat_dict

X,y,c=loadimgs(train_folder)


with open(os.path.join(save_path,"train.pickle"), "wb") as f:
	pickle.dump((X,c),f)


X,y,c=loadimgs(valpath)
with open(os.path.join(save_path,"test.pickle"), "wb") as f:
	pickle.dump((X,c),f)