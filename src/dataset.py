import os
import xmltodict
import pandas as pd

def read_dataset(data_dir):
    data_dir = os.path.realpath(data_dir)

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if file_path.endswith(".xml"):

                with open(file_path) as fd:
                    data = xmltodict.parse(fd.read())
                    
                data = data["annotation"]
                data["path"] = file_path.replace(".xml", ".jpg")

                if "object" in data:
                    objects = data["object"]

                    if isinstance(data["object"], dict):
                        objects = [objects]

                    for obj in objects:
                        data_ = data.copy()
                        data_.update(data_["size"])
                        data_.update(obj)
                        data_.update(obj["bndbox"])

                        yield data_

def get_base_df(data_dir):
    generator = read_dataset(data_dir)
    df = pd.DataFrame(generator)

    return df

