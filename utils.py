
import os
import shutil
import random
import re
import numpy as np
import cv2


def sort_by_file_name(dirnames):
    dirs = sorted(dirnames, key=lambda x: (int(re.sub('\D', '', x)), x))
    return dirs


def read_img(path, target_img_size):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, target_img_size, cv2.INTER_LINEAR)
    return img


def load_fold_data(fold, target_img_size, fold_classes=True):
    x_test = []
    y_test = []
    file_test = []
    _dirs = os.listdir(fold)
    dirs = sort_by_file_name(_dirs)
    print('find ' + str(len(dirs)) + ' from ' + fold)
    i = 0

    for _dir in dirs:
        files = os.listdir(fold + "/" + _dir)

        for file in files:
            img_path = fold + "/" + _dir + "/" + file
            if os.path.isfile(img_path):
                try:
                    img = read_img(img_path, target_img_size)
                    # for 4 channels imgs
                    if img.shape[2] > 3:
                        img = img[:,:,0:3]

                    file_test.append(img_path)
                    if fold_classes:
                        y_test.append(int(_dir))
                    else:
                        y_test.append(int(i))
                    x_test.append(img)
                except:
                    print('error img ' + img_path)
        i += 1

    print('load data from fold finished ,count=' + str(len(y_test)))
    return x_test, y_test, file_test


def split_train_valid_fold(all_path, train_path, valid_path, split_ratio=0):
    shutil.rmtree(train_path, ignore_errors=True)
    shutil.rmtree(valid_path, ignore_errors=True)

    _files = os.listdir(all_path)
    files = sort_by_file_name(_files)

    for file in files:
        class_files = os.listdir(all_path + '/' + file)

        if not os.path.exists(valid_path + '/' + file):
            os.makedirs(valid_path + '/' + file)
        if not os.path.exists(train_path + '/' + file):
            os.makedirs(train_path + '/' + file)

        random.shuffle(class_files)
        num = round(len(class_files) * split_ratio) if split_ratio < 1 else split_ratio
        assert (num < len(class_files))

        for i in range(num):
            shutil.copy2(all_path + '/' + file + '/' + class_files[i], valid_path + '/' + file)

        for i in range(num, len(class_files)):
            shutil.copy2(all_path + '/' + file + '/' + class_files[i], train_path + '/' + file)

    print('split train and valid fold finished, all_path: %s, split_ratio: %s' % (all_path, split_ratio))

