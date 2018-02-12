# Object Detection
## Introduction
*TODO* 

#### Requirements 
Please install the following tools:
* docker
* docker-compose
* [nvidia-docker-2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))

Also configure `nvidia-docker` to work with `docker-compose`.

#### Getting the Code
Please clone the repo and `cd` into it.
```
git clone https://github.com/cgarciae/object-detection.git
cd object-detection
```

#### Getting the Data
First create a `data` directory inside the repo's folder. This directory won't be tracked by git. Now, download + extract the data from the google drive folder here and rename the folder to `base`. You should end with a structure like this:
```
+ data
  + base
    + movie1
    + movie2
    ...
```
#### Build container
First build the container to get the environment ready with
```
docker-compose build
```

#### Create Protos
Finally, we have to compile some protobufs that google doesn't include by default. Just execute
```
docker-compose run jupyter bash scripts/create_protos.sh
```

## Data 

#### Visualization
A jupyter notebook was created to explore the data a little. You open the notebook by running the commands:
```
docker-compose up -d
docker-compose logs -f
```
After jupyter start it should print a link, copy the link from the console to a web browser. On jupyter navigate to the notebooks folder and open `data-exploration.ipynb`. You will see 2 main visualizations:
* A graph showing the distribution of labels
* The images resulting from data augmentation (discussed next)

#### Data Augmentation

Data augmentation was used with the idea of helping the model to generalize better by broadening the distribution of the data with various types of random filters. The library [imgaug](https://github.com/aleju/imgaug) was used for this purpose. The following augmenter where used:

```python
iaa.Grayscale(alpha=(0.0, 1.0)),
iaa.GaussianBlur(sigma=(0.0, 3.0)),
iaa.AverageBlur(k=(2, 9)),
iaa.MedianBlur(k=(3, 9)),
iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
iaa.Add((-40, 40), per_channel=0.5),
iaa.AddElementwise((-40, 40), per_channel=0.5),
iaa.AdditiveGaussianNoise(scale=0.05*255, per_channel=0.5),
iaa.Multiply((0.5, 1.5), per_channel=0.5),
iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5),
iaa.Dropout(p=(0, 0.2), per_channel=0.5),
iaa.CoarseDropout(0.05, size_percent=0.1),
iaa.Invert(1.0, per_channel=0.5),
iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25),
iaa.PiecewiseAffine(scale=(0.01, 0.05)),
```
The resulting randomized images look like this

![data-augmentation](readme/data-augmentation.png "data-augmentation")


#### Discussion on Data Augmentation
Data Augmentation generally improves the performace of a model as it forces it to generalize more since it can't depend on any specific details being present in the image but rather it has to abstract more the entities its trying to recognize. However, since we are doing transfer learning with fine tunning by using the last layer of a pretrained model, this layer might already be good enough and data augmentation will have less effect. 

Nevertheless having more data is always better and the variations will still help.

#### Convert data to tfrecords
The `object_detection` module already provides various functions for converting Pascal VOC datasets (like the one privided) to tfrecords, especifically, the script `create_pascal_tf_record.py` lets you convert a whole folder of samples to this format. However, since it was decided that data augmentation was to be used, various scripts like `src/dataset.py` and `src/preprocessing.py` where created to do various tasks like:
* Converting XML data to Python dictionaries
* Doing data augmentation
* Converting the augmented data to `tf.train.Example`s and writting these to disk.

The development time took longer due to this extra feature. To convert the data just run the command to tfrecords just run the commands:

```bash
# training set
docker-compose run jupyter python3 src/cli.py create_data training

# test set
docker-compose run jupyter python3 src/cli.py create_data test
```
This creates the files `data/training_set.record` and `data/test_set.record`. Theses operations might take a while to complete.

## Model
With the `object_detection` module you can select from a wide variety of pretrained models that you can checkout in their [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models). As you can see they varie in speed and accuracy (as mesured in the COCO dataset via the `mAP[^1]` metric), `ssd_inception_v2_coco` was selected for this excercise. 

#### Download
To download the model just run
```
docker-compose run jupyter bash scripts/download_model.sh
```
this should create the `models` folder and download the tensorflow checkpoints from `ssd_inception_v2_coco` inside.

## Train & Eval
#### Config
There are 2 files which configure the training and evaluation of our model:
* `ssd_inception_v2.config`
* `label_map.pbtxt`

The first one is very complex and lets you specify varios aspects of the training, it was taken from the `object_detection/samples/configs` and modified according to the structure of our project. The second one just specifies the indexes for each category.

#### Train
Run
```bash
docker-compose run jupyter bash scripts/train.sh
```
This might take a while, you possibly need a machine with > 4GB of GPU RAM.

## Evaluate
Run
```bash
docker-compose run jupyter bash scripts/eval.sh
```
This might take a while.
