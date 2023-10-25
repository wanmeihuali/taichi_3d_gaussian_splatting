import argparse
import yaml
from pathlib import Path

parser = argparse.ArgumentParser("Prepare training for 3D Gaussian Splatting")
parser.add_argument("--example_config", type=str, required=True, help="Path to example config yaml file")
parser.add_argument("--input_prefix", type=str, required=True, help="Path prefix to train.json,val.json,point_cloud.parquet")
args=parser.parse_args()
with open(args.example_config, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    input_prefix = Path(args.input_prefix)
    config['train-dataset-json-path']=str(input_prefix / "train.json")
    config['val-dataset-json-path']=str(input_prefix / "val.json")
    config['pointcloud-parquet-path']=str(input_prefix / "point_cloud.parquet")
    config['summary-writer-log-dir']=args.input_prefix
    config['output-model-dir']=args.input_prefix
    with open("train.yaml", "w+") as w:
        yaml.dump(config, w)
