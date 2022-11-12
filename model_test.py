import sys
import yaml
import torch
# from torch.utils.data import DataLoader

# import module from the same directory
from model.model import MusicCLIP
from dataloader.dataloader_updated import get_dataloader

from model.train_new import train
from config.text_config import text_args

print("loading config...")
config_path = "config/default.yaml"

global trained_steps

# config = MusicConfig(config_path)

recons_loss_ema = 0.
kl_loss_ema = 0.
kl_raw_ema = 0.

# for testing purpose just hardcode this value
vocab_size = 333

def test_dataloader(config):
    train_dset, val_dset, test_dset, train_dloader, val_dloader, test_dloader = get_dataloader(config)
    print("Printing example of train_dloader output...")
    for i, batch in enumerate(train_dloader):
        for k, v in batch.items():
            if torch.is_tensor(v):
                print (k, ':', v.dtype, v.size())
            else:
                print(k, ':', v)
            break
        print ('=====================================\n')
        if i == 5: break

    return train_dset, val_dset, test_dset, train_dloader, val_dloader, test_dloader


if __name__ == "__main__":

    # test dataloader
    # print("testing dataloader...")
    data_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    train_dset, val_dset, test_dset, train_dloader, val_dloader, test_dloader = test_dataloader(data_config)
    
    model = MusicCLIP(data_config, text_args)
    # print(model.state_dict().keys())

    train(music_config = data_config, text_config = text_args)
