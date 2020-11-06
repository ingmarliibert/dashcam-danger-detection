import pathlib
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import time
import pandas as pd
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2

batch_size = 20
img_height = 32
img_width = 32

TRAFFIC_SIGN_ROOT = './data/traffic-sign'
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
  pathlib.Path(TEST_CLEAN),
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# For performance
# https://www.tensorflow.org/tutorials/images/classification#configure_the_dataset_for_performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

"""
@TODO: https://www.tensorflow.org/tutorials/images/data_augmentation
"""
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
train_normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_normalized_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

NUM_CLASSES = 43

# Hyper-parameters
LEARNING_RATE = 4e-4
EPOCHS = 20
# BATCH_SIZE = 55

# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Prepare the metrics for train & test
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


def tf_mobilenetv2():
    """
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2
    """
    from tensorflow.keras.layers import Input
    model = MobileNetV2(classes=NUM_CLASSES, weights=None, input_tensor=Input(shape=(img_height, img_width, 3)))

    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, batch_size=batch_size)

    return model

# model = tf_mobilenetv2()
# model.save('saved_model/mobilenetv2')

model = tf.saved_model.load('saved_model/mobilenetv2')

def test():
    img = keras.preprocessing.image.load_img(
        './data/traffic-sign/test-clean/1/00792.png', target_size=(32, 32)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
    image_array = normalization_layer(img_array)

    img_array = tf.expand_dims(img_array, 0) # Create a batch

    print(f'img shape: {img_array.shape}')

    #predictions = model.predict(img_array)
    predictions = model(img_array, training=False)
    print(predictions)

    sign = pd.read_csv('signnames.csv')
    sign_names = sign['SignName']

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(sign_names[np.argmax(predictions)], 100 * np.max(predictions))
    )

test()
