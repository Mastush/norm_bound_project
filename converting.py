import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import gray2rgb, rgba2rgb
from skimage import img_as_ubyte
import time

import cv2


def method_1():
    source_dir = '/mnt/local-hd/nadavsch/ImageNet2012/val'
    target_dir = '/mnt/local-hd/nadavsch/ImageNet2012/vaLprocessed'
    i = 0
    start_time = time.time()
    subdir_list = os.listdir(source_dir)
    for subdir in subdir_list:
        os.mkdir(os.path.join(target_dir, subdir))
        subdir_start_time = time.time()
        for im_name in os.listdir(os.path.join(source_dir, subdir)):
            im_path = os.path.join(source_dir, subdir, im_name)
            im = imread(im_path)
            if len(im.shape) < 3:  # grayscale
                im = gray2rgb(im)
            elif im.shape[2] == 4:  # rgba
                im = rgba2rgb(im)
            im = resize(im, (224, 224, 3))
            im = img_as_ubyte(im)
            imsave(os.path.join(target_dir, subdir, im_name), im)
        subdir_time = int(time.time() - subdir_start_time) / 60
        total_time = int(time.time() - start_time) / 60
        i += 1
        print("Finished subdir {} out of {}. This took {:.2f} minutes. "
              "Total time {:.2f} minutes.".format(i, len(subdir_list), subdir_time, total_time))


def method_2():
    source_dir = '/mnt/local-hd/nadavsch/ImageNet2012/train'
    target_dir = '/mnt/local-hd/nadavsch/ImageNet2012/train_processed'
    i = 0
    start_time = time.time()
    subdir_list = os.listdir(source_dir)
    for subdir in subdir_list:
        if not os.path.exists(os.path.join(target_dir, subdir)):
            os.mkdir(os.path.join(target_dir, subdir))
        subdir_start_time = time.time()
        for im_name in os.listdir(os.path.join(source_dir, subdir)):
            im_path = os.path.join(source_dir, subdir, im_name)
            im = cv2.imread(im_path)
            if len(im.shape) < 3:  # grayscale
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            elif im.shape[2] == 4:  # rgba
                im = cv2.cvtColor(im , cv2.COLOR_RGBA2RGB)
            im = cv2.resize(im, (224, 224))
            # im = img_as_ubyte(im)
            cv2.imwrite(os.path.join(target_dir, subdir, im_name), im)
        subdir_time = int(time.time() - subdir_start_time) / 60
        total_time = int(time.time() - start_time) / 60
        i += 1
        print("Finished subdir {} out of {}. This took {:.2f} minutes. "
              "Total time {:.2f} minutes.".format(i, len(subdir_list), subdir_time, total_time))


