import os
import pickle
import math
import random
import csv
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import time

from traffic_sign_models.google_le_net import GoogleNet

# 1 Reload the preprocessed data

ROOT_DATA = 'data/traffic-sign-pickled'

pickle_file = f'{ROOT_DATA}/pre-data.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    X_train = pickle_data['train_features']
    y_train = pickle_data['train_labels']
    X_valid = pickle_data['valid_features']
    y_valid = pickle_data['valid_labels']
    X_test = pickle_data['test_features']
    y_test = pickle_data['test_labels']
    signnames = pickle_data['signnames']
    del pickle_data  # Free up memory

# Shuffle the data set
X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)
X_test, y_test = shuffle(X_test, y_test)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)
print(len(signnames))
print('Data loaded.')

# Training

## Strategy

# Placeholder
# fixes: RuntimeError: tf.placeholder() is not compatible with eager execution.
# tf.compat.v1.disable_eager_execution()

# x = tf.compat.v1.placeholder(tf.float32, (None, 32, 32, 3))
# y = tf.compat.v1.placeholder(tf.int32, (None))
# one_hot_y = tf.one_hot(y, 43)
# keep_prob = tf.compat.v1.placeholder_with_default(1.0, shape=())

NUM_CLASSES = 43
model = GoogleNet(NUM_CLASSES)

# Hyper-parameters
LEARNING_RATE = 4e-4
EPOCHS = 35
BATCH_SIZE = 55

# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


@tf.function
def train_step(images, labels, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # train_loss(loss)
    # train_accuracy(labels, predictions)

    return loss


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


# X_train, y_train = shuffle(X_train, y_train)

"""
The data has to be of shape 4.
More information here: https://datascience.stackexchange.com/questions/64072/input-0-is-incompatible-with-layer-conv2d-2-expected-ndim-4-found-ndim-3-i-get
Basically, images need to be 4D [batch_size, img_height, img_width, number_of_channels]
This is done by tensorflow here.
"""
train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).shuffle(10000).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)).batch(BATCH_SIZE)

"""
Prefer tf.train.Checkpoint over save_weights for training checkpoints.
from: https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights
So -> https://www.tensorflow.org/guide/checkpoint#manual_checkpointing
"""
iterator = iter(train_ds)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model, iterator=iterator)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    step = 0
    for (images, labels) in iterator:
        # Log every 200 batches.
        if step % 200 == 0:
            print("Seen so far: %d samples" % ((step + 1) * 64))

        loss = train_step(images, labels, optimizer)
        ckpt.step.assign_add(1)
        step = step+1

        if int(ckpt.step) % 10 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("loss {:1.2f}".format(loss.numpy()))

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    # Display metrics at the end of each epoch.
    template = 'Epoch: [{}/{}], Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          EPOCHS,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))
