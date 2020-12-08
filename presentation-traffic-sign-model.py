import pathlib

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

matplotlib.use('TkAgg')

from model.traffic_sign import traffic_sign_factory

img_height = 50
img_width = 50


batch_size = 5

TRAIN_ROOT = './data/traffic-sign/train'
TEST_CLEAN = './data/traffic-sign/test-clean'


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    pathlib.Path(TRAIN_ROOT),
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    labels='inferred',
    label_mode='int')


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    pathlib.Path(TRAIN_ROOT),
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

traffic_sign = traffic_sign_factory()

num = 3

for i in range(num):

    plt.figure(figsize=(10, 10))
    for images, labels in val_ds.take(1):
      for i in range(batch_size-1):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        result = traffic_sign.inception(images[i], False)

        print(result)
        plt.title(result.class_name)
        plt.axis("off")

    plt.show()


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
