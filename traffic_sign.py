import os

import cv2
import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


TRAFFIC_SIGN_ROOT = './data/traffic-sign'
TRAIN_ROOT = './data/traffic-sign/Train'

# function to read and resize images, get labels and store them into np array
def get_image_label_resize(label, files, dim=(32, 32), dataset='Train'):
    def clean_image(img_path):
        img = cv2.resize(cv2.imread(img_path), dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        norm = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
        return norm

    x = np.array([clean_image(fname) for fname in files])
    y = np.array([label] * len(files))

    return x, y

def get_train():
    x_train = None
    y_train = None

    # go through all labels and store images into np arrays
    for label in range(0, 43):
        x, y = get_image_label_resize(label, glob.glob(f'{TRAIN_ROOT}/{label}/*.png'))
        print(x.shape)

        if x_train is None:
            x_train = x
            y_train = y
        else:
            x_train = np.concatenate((x_train, x))
            y_train = np.concatenate((y_train, y))

    return x_train, y_train

pickle_file = './pre-data.pickle'

if not os.path.isfile(pickle_file):
    print('Process & save data to pickle file...')
    x_train, y_train = get_train()
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'x_train': x_train,
                    'y_train': y_train,
                },
                pfile, protocol=2)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
else:
    with open(pickle_file, mode='rb') as f:
        data = pickle.load(f)
        x_train = data['x_train']
        y_train = data['y_train']

test_df = pd.read_csv(f'{TRAFFIC_SIGN_ROOT}/Test.csv')
# get path for test images
testfile = test_df['Path'].apply(lambda x: f'{TRAFFIC_SIGN_ROOT}/{x}').tolist()

X_test = np.array([cv2.resize(cv2.imread(file_path), (32, 32), interpolation = cv2.INTER_AREA) for file_path in testfile])
y_test = np.array(test_df['ClassId'])

# shuffle training data and split them into training and validation

indices = np.random.permutation(x_train.shape[0])

# 20% to val, int is used to round.
split_idx = int(x_train.shape[0]*0.8)
train_idx, val_idx = indices[:split_idx], indices[split_idx:]
X_train, X_validation = x_train[train_idx,:], x_train[val_idx,:]
y_train, y_validation = x_train[train_idx], x_train[val_idx]

# # get overall stat of the whole dataset
n_train = X_train.shape[0]
n_validation = X_validation.shape[0]
n_test = X_test.shape[0]
image_shape = X_train[0].shape
n_classes = len(np.unique(y_train))
print("There are {} training examples ".format(n_train))
print("There are {} validation examples".format(n_validation))
print("There are {} testing examples".format(n_test))
print("Image data shape is {}".format(image_shape))
print("There are {} classes".format(n_classes))
