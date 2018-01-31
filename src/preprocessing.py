from dicto import dicto
import pandas as pd
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import utils
import tensorflow as tf
import hashlib
from io import BytesIO
import os

from python_path import PythonPath
with PythonPath(relative = "."):
    from object_detection.utils import dataset_util


# def balance_dataset(df, params = None):
#     if params is None:
#         params = dicto.load_("params.yml")

#     # get parameters
#     samples_per_group = params.preprocessing.samples_per_group

#     # dataframes list
#     dfs = [] 

#     # group data by name
#     groups = df.groupby("name")

#     # iterate over groups
#     for name, dfg in groups:
#         # if samples wanted are greater than existing samples
#         if samples_per_group > len(dfg):
#             # calculate the number of remaining samples
#             samples = samples_per_group - len(dfg)
#             # and sample from the group with replacement
#             dfg_sample = dfg.sample(samples, replace = True)
#             # then add the extra samples to the data
#             dfg = pd.concat([dfg, dfg_sample])
#         else:
#             # if not, just sample from the group data
#             dfg = dfg.sample(samples_per_group)
        
#         # finally append the group data to the list
#         dfs.append(dfg)

#     # and concatenate all groups
#     df = pd.concat(dfs)

#     return df


def augmentation_sequence(params):
    if params is None:
        params = dicto.load_("params.yml")

    
    n_augmenters = params.data_augmentation.n_augmenters

    return iaa.SomeOf(n_augmenters, [
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
    ])

def load_image(filename):
    with Image.open(filename) as img:
        return np.asarray(img)


def read_images(generator):

    for batch in generator:

        batch["image"] = batch.path.apply(load_image)

        yield batch


def augment_images(df, params = None):
    if params is None:
        params = dicto.load_("params.yml")
    
    # get augmentation sequence
    seq = augmentation_sequence(params)

    df["image"] = df.image.apply(seq.augment_image)

    return df
    
# create additional samples
def grow_dataset(df, params = None):
    if params is None:
        params = dicto.load_("params.yml")
    
    return pd.concat( [ df ] * params.data_augmentation.n_copies).reset_index()
    

def data_augmentation(generator, params = None):
    if params is None:
        params = dicto.load_("params.yml")


    for batch in generator:

        batch = augment_images(batch, params = params)

        yield batch


def create_tfrecords(generator, set_name, params):

    output_path = os.path.join(params.preprocessing.output_path, set_name)
    writer = tf.python_io.TFRecordWriter(output_path)

    for batch in generator:
        for i, row in batch.iterrows():
            example = create_example(row, params)
            writer.write(example.SerializeToString())

    writer.close()



def create_example(data, params):

    imgByteArr = BytesIO()
    img = Image.fromarray(data.image)
    img.save(imgByteArr, 'JPEG')
    encoded_jpg = imgByteArr.getvalue()
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    if not isinstance(data['object'], list):
        data['object'] = [ data['object'] ]

    for obj in data['object']:
        try:
            difficult = bool(int(obj['difficult']))
        except:
            print(obj)
            print(data)
            continue
        
        
        if params.preprocessing.ignore_difficult_instances and difficult:
            continue

        difficult_obj.append(int(difficult))
        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(params.label_map_dict[obj['name']])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))

    return tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))