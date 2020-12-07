# Dash cam danger detection

## Requirements
Python >= 3.8 (and maybe >=3.7 could also works but I'm not sure).

Install them, for example via pip in a terminal:
```
pip install --requirement ./requirements.txt
```

Or any virtualenv or something that you want.

if you have trouble installing scikit with pycharm :
```
python -m pip install scikit-learn==0.23.2
```

## Data
Install the [Traffic sign dataset](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign), unzip and put it in the folder `./data/traffic-sign`.


## Tensorflow
Follow instructions here:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

The base tutorial: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb

Make sure that you have Tensorflow 2 installed, Tensorflow 1 won't work.
If have followed the previous step, then it should be fine.

## Models and Data Provenance

### Coco-SSD
- Description
    - This model uses 80 categories from the COCO image dataset to detect objects.
- References
    - [Tensorflow Object Detection API.](https://github.com/tensorflow/models/tree/master/research/object_detection)
    - Paper by Microsoft: COCO: [Common Objects in Context](https://arxiv.org/abs/1405.0312)    
    - [COCO website.](http://cocodataset.org/#home)
- Sources
    - We use a detection model pre-trained on the COCO 2017 dataset by Tensorflow. [You can find a list of models here.](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
    - We use the SSD mobilenet v2 coco17 because only SSD models are supported by Tensorflow Lite, which could be handy in the future and this is a very performant model.

### Traffic sign
We trained a model to recognize traffic signs. The model is a MobilenetV2 trained from scratch.
That means we didn't use transfer learning based on the COCO dataset (or anything else).

- Description
    - This model uses 43 traffic sign categories from the GTSRB dataset.
- References
    - Tensorflow 2.
    - MobilenetV2 model.
- Sources
    - You can find the dataset [here](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
    - You can find labels [here](./signnames.csv). It is a csv file that map a number (used by the model) to a class name, used by humans.
    - You can find the trained model [here](./saved_model).
