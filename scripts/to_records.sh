export PYTHONPATH=PYTHONPATH:$PWD:$PWD/slim

python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=./data/base \
        --label_map_path=./data/base_label_map.pbtxt \
        --output_path=./data/base.record