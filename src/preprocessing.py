from dicto import dicto
import pandas as pd
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import utils

def preprocess_dataset(df, params = None):
    if params is None:
        params = dicto.load_("params.yml")

    df = df[df.name != "car"]
    df = df[df.name != "hie"]

    return df

def balance_dataset(df, params = None):
    if params is None:
        params = dicto.load_("params.yml")

    # get parameters
    samples_per_group = params.preprocessing.samples_per_group

    # dataframes list
    dfs = [] 

    # group data by name
    groups = df.groupby("name")

    # iterate over groups
    for name, dfg in groups:
        # if samples wanted are greater than existing samples
        if samples_per_group > len(dfg):
            # calculate the number of remaining samples
            samples = samples_per_group - len(dfg)
            # and sample from the group with replacement
            dfg_sample = dfg.sample(samples, replace = True)
            # then add the extra samples to the data
            dfg = pd.concat([dfg, dfg_sample])
        else:
            # if not, just sample from the group data
            dfg = dfg.sample(samples_per_group)
        
        # finally append the group data to the list
        dfs.append(dfg)

    # and concatenate all groups
    df = pd.concat(dfs)

    return df


def augmentation_sequence(params):
    if params is None:
        params = dicto.load_("params.yml")

    
    n_augmenters = params.data_augmentation.n_augmenters

    return iaa.SomeOf(n_augmenters, [
        iaa.Superpixels(p_replace=0.5, n_segments=64),
        iaa.Grayscale(alpha=(0.0, 1.0)),
        iaa.GaussianBlur(sigma=(0.0, 2.0)),
        iaa.AverageBlur(k=(2, 7)),
        iaa.MedianBlur(k=(3, 7)),
        iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
        iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
        iaa.Add((-40, 40), per_channel=0.5),
        iaa.AddElementwise((-40, 40), per_channel=0.5),
        iaa.AdditiveGaussianNoise(scale=0.05*255, per_channel=0.5),
        iaa.Multiply((0.5, 1.5), per_channel=0.5),
        iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5),
        iaa.Dropout(p=(0, 0.2), per_channel=0.5),
        iaa.CoarseDropout(0.02, size_percent=0.2, per_channel=0.5),
        iaa.Invert(1.0, per_channel=0.5),
        iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
        iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25),
        iaa.PiecewiseAffine(scale=(0.01, 0.05)),
    ])

def load_image(filename):
    with Image.open(filename) as img:
        return np.asarray(img)


def augment_images(df, params = None):
    if params is None:
        params = dicto.load_("params.yml")

    # maybe load images
    if "image" not in df:
        df["image"] = df.path.apply(load_image)
    
    # get augmentation sequence
    seq = augmentation_sequence(params)

    df["image"] = df.image.apply(seq.augment_image)

    return df
    

def data_augmentation(df, params = None):
    if params is None:
        params = dicto.load_("params.yml")
    

    for batch in utils.dataframe_batch_generator(df, params.data_augmentation.batch_size):

        batch = augment_images(batch, params = params)

        yield batch
