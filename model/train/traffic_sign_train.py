import json
import pathlib
import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Input
import numpy as np

# from model.train.consts import img_height, img_width

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

# plt.figure(figsize=(10, 10))
# for images, labels in val_ds.take(1):
#   for i in range(batch_size-1):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
#
# plt.show()

# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break


# For performance
# https://www.tensorflow.org/tutorials/images/classification#configure_the_dataset_for_performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

"""
@TODO: https://www.tensorflow.org/tutorials/images/data_augmentation
"""
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

# train_normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# val_normalized_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


NUM_CLASSES = 43

# Hyper-parameters
# LEARNING_RATE = 4e-4
EPOCHS = 1
# BATCH_SIZE = 55

# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
# learning_rate = LEARNING_RATE
optimizer = tf.keras.optimizers.Adam()
# optimizer = tf.keras.optimizers.SGD()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()


def tf_mobilenetv2():
    """
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2
    """
    model = MobileNetV2(classes=NUM_CLASSES, weights=None, input_tensor=Input(shape=(img_height, img_width, 3)))

    log_dir = "logs/fit-mobilenet-test2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        batch_size=batch_size,
        callbacks=[tensorboard_callback])

    return model


def personal():
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(img_height,
                                                                      img_width,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        # data_augmentation,
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        # layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])

    log_dir = "logs/fit-personal/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[tensorboard_callback]
    )

    return model


model = tf_mobilenetv2()
model.save('saved_model/mobilenetv2-test2')

with open('saved_model/mobilenetv2-test2/classes.json', 'w') as f:
    json.dump(class_names, f)

#
# model = tf.keras.models.load_model('saved_model/mobilenetv2-test2')
#
# # model = personal()
# # model.save('saved_model/personal')
#
# img1 = keras.preprocessing.image.load_img('./data/traffic-sign/test-clean/2/00067.png', target_size=(img_height, img_width), interpolation='bilinear')
# img2 = keras.preprocessing.image.load_img('./data/traffic-sign/test-clean/3/00036.png', target_size=(img_height, img_width), interpolation='bilinear')
# img3 = keras.preprocessing.image.load_img('./data/traffic-sign/test-clean/15/00210.png', target_size=(img_height, img_width), interpolation='bilinear')
# img4 = keras.preprocessing.image.load_img('./data/traffic-sign/test-clean/17/00035.png', target_size=(img_height, img_width), interpolation='bilinear')
# img5 = keras.preprocessing.image.load_img('./data/traffic-sign/test-clean/34/00386.png', target_size=(img_height, img_width), interpolation='bilinear')
#
#
# def process(img):
#     img_array = keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)  # Create a batch
#
#     predictions = model.predict(img_array)
#     score = tf.nn.softmax(predictions[0])
#
#     print(
#         "This image most likely belongs to {} with a {:.2f} percent confidence."
#             .format(class_names[np.argmax(score)], 100 * np.max(score))
#     )
#
# process(img1)
# process(img2)
# process(img3)
# process(img4)
# process(img5)
#
# print('evaluation after load', model.evaluate(x=train_ds, y=val_ds))
