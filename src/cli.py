from python_path import PythonPath


import click
import dataset
import preprocessing as pr
import utils
from dicto import dicto


@click.group()
def main():
    pass


@main.command()
@click.argument("set_name")
def create_data(set_name):
    

    if set_name == "training" or set_name == "test":
        params = dicto.load_("params.yml")

        df = dataset.get_base_df("data/base")

        split = int(len(df) *  params.preprocessing.split)
        training_df = df.iloc[:split]
        test_df = df.iloc[split:]
        

        if set_name == "training":
            df = training_df
            df = pr.grow_dataset(df, params = params)

            # df = df.iloc[:100]								
            
            generator = utils.dataframe_batch_generator(df, params.preprocessing.batch_size)
            generator = pr.read_images(generator)
            generator = pr.data_augmentation(generator, params = params)

            pr.create_tfrecords(generator, "training_set.record", params)
        else:
            df = test_df
            
            
            generator = utils.dataframe_batch_generator(df, params.preprocessing.batch_size)
            generator = pr.read_images(generator)

            pr.create_tfrecords(generator, "training_set.record", params)
    else:
        raise Exception("Please choose 'training' or 'test', got {set_name}".format(set_name = set_name))


if __name__ == '__main__':
    main()