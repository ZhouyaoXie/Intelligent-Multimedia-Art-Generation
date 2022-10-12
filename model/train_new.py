import os
import time
import json
import copy
import pickle
import random
import requests
import numpy as np
from PIL import Image
import re

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from config.text_config import text_args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("the set device is ", device)

#setting the seeds
GLOBAL_SEED = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True




def dataloader(config):
    # load MuseMorphose REMI dataset for testing
    dset = REMIFullSongTransformerDataset(
        config['data']['data_dir'], config['data']['vocab_path'],
        do_augment=True,
        model_enc_seqlen=config['data']['enc_seqlen'],
        model_dec_seqlen=config['data']['dec_seqlen'],
        model_max_bars=config['data']['max_bars'],
        pieces=pickle_load(config['data']['train_split']),
        pad_to_same=True
    )
    dset_val = REMIFullSongTransformerDataset(
        config['data']['data_dir'], config['data']['vocab_path'],
        do_augment=False,
        model_enc_seqlen=config['data']['enc_seqlen'],
        model_dec_seqlen=config['data']['dec_seqlen'],
        model_max_bars=config['data']['max_bars'],
        pieces=pickle_load(config['data']['val_split']),
        pad_to_same=True
    )
    print('[info]', '# training samples:', len(dset.pieces))

    dloader = DataLoader(
        dset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8)
    dloader_val = DataLoader(
        dset_val, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8)

    return dset, dset_val, dloader, dloader_val


music_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)


def _train():
    trainset, trainset_val, train_loader, train_loader_val = dataloader(data_config)

    # # need to get from true dataloader 
    # config.n_token = 333

    model = MusicCLIP(music_config, text_config)
    print(model.state_dict().keys())

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 5e-4)

    epochs = data_config['training']['max_epochs']
    model.zero_grad()
	# model = model.double()
	start_time  = time.time()
	model.train()
	for epoch in range(epochs):
		print("Starting epoch ", epoch)
		for step, (x,y) in enumerate(train_loader):
		    x = x.to(device)
		    y = y.to(device)
		    lang_feats, vision_feats, pooled_output = model(x)
		    loss = autoencoder_loss(d,x)
		    optimizer.zero_grad()
		    loss.backward()
		    optimizer.step()
			print("loss", loss.item())
		    #training accuracy



def train():
    mode = params.mode.lower()
    if mode == "TRAIN":
        _train()
    elif mode == "EVAL":
        _eval()
    else:
        raise ValueError("Unrecognized mode!")