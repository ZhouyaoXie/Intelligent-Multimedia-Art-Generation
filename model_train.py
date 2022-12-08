import sys
import yaml
# import torch
import argparse
import numpy as np
import tqdm

# import module from the same directory
from model.model import MusicCLIP
from dataloader.dataloader_updated import get_dataloader

from model.train_new import train
from config.text_config import text_args

from model.remi2midi import remi2midi


print("loading config...")
config_path = "config/default.yaml"

# @utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='model_train.py')
#     parser.add_argument('--output_path', type=str, required=True, help="Output path.")
#     parser.add_argument('--mode', type=str, required=True, help="TRAIN or INFERENCE.")
    parser.add_argument('--output_path', type=str, default="outputs/", help="Output path.")
    parser.add_argument('--mode', type=str, default="TRAIN", help="TRAIN or INFERENCE.")
    parser.add_argument('--max_lr', type=float, default=1.0e-4, help="Max learning rate.")
    parser.add_argument('--min_lr', type=float, default=5.0e-6, help="Min learning rate.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size.")
    parser.add_argument('--max_epochs', type=int, default=1000, help="Max training epochs.")
    return parser.parse_args(args=args, namespace=namespace)

if __name__ == "__main__":

    # train
    print("training . . .")
    data_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
#     args = parse_args()

#     data_config["data"]["batch_size"] = args.batch_size
#     data_config["training"]["output_path"] = args.output_path
#     data_config["training"]["max_epochs"] = args.max_epochs
#     data_config["training"]["max_lr"] = args.max_lr
#     data_config["training"]["min_lr"] = args.min_lr
#     data_config["training"]["mode"] = args.mode

#     print("mode: ", data_config["training"]["mode"])
#     print("output_path: ", data_config["training"]["output_path"])
#     print("max_epochs: ", data_config["training"]["max_epochs"])
#     print("bastch_size: ", data_config["data"]["batch_size"])
#     print("max_lr: ", data_config["training"]["max_lr"])
#     print("min_lr: ", data_config["training"]["min_lr"])
    
    # train(music_config = data_config, text_config = text_args)
    with open("outputs/id0_polyNone_rhymNone.txt", 'r') as f:
        out_file = f.readlines()
    song = [s.strip() for s in out_file]
    remi2midi(song, 'id0_polyNone_rhymNone.mid', enforce_tempo=True)
