export PYTHONPATH=PYTHONPATH:$PWD:$PWD/slim

protoc object_detection/protos/*.proto --python_out=.