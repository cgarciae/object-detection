import os
import xmltodict
import pandas as pd

def read_dataset(data_dir):
    data_dir = os.path.realpath(data_dir)

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)

        for filename in os.listdir(folder_path):
            
            if filename.endswith(".xml"):
                
                file_path = os.path.join(folder_path, filename)

                with open(file_path) as fd:
                    data = xmltodict.parse(fd.read())

                jpg_file = filename.replace(".xml", ".jpg")
                jpg_filepath = file_path.replace(".xml", ".jpg")
                    
                data = data["annotation"]
                data["path"] = jpg_filepath
                data["id"] = folder + "_" + jpg_file

                data["n_wheels"] = 0
                data["n_hoe"] = 0
                data["n_body"] = 0

                if "object" not in data:
                    continue
                elif isinstance(data["object"], dict):
                    objs = [data["object"]]
                elif isinstance(data["object"], list):
                    objs = data["object"]

                for obj in objs:
                    if obj["name"] == "wheels":
                        data["n_wheels"] += 1
                    elif obj["name"] == "body":
                        data["n_body"] += 1
                    elif obj["name"] == "hoe":
                        data["n_hoe"] += 1

                objs = filter(lambda o: o["name"] in ["wheels", "hoe", "body"], objs)

                data["object"] = list(objs)

                yield data


                

def get_base_df(data_dir):
    generator = read_dataset(data_dir)
    df = pd.DataFrame(generator)

    return df

