import os
import time
import json
import copy
import pickle
import random
import numpy as np
import yaml 
import re

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model.model import MusicCLIP

from dataloader.dataloader_updated import get_dataloader

from config.text_args import text_config
from .contrastive_loss import ContrastiveLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("the set device is ", device)

#setting the seeds
GLOBAL_SEED = 1
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

config_path = "config/default.yaml"

# def dataloader(config):
#     # load MuseMorphose REMI dataset for testing
#     dset = REMIFullSongTransformerDataset(
#         config['data']['data_dir'], config['data']['vocab_path'],
#         do_augment=True,
#         model_enc_seqlen=config['data']['enc_seqlen'],
#         model_dec_seqlen=config['data']['dec_seqlen'],
#         model_max_bars=config['data']['max_bars'],
#         pieces=pickle_load(config['data']['train_split']),
#         pad_to_same=True
#     )
#     dset_val = REMIFullSongTransformerDataset(
#         config['data']['data_dir'], config['data']['vocab_path'],
#         do_augment=False,
#         model_enc_seqlen=config['data']['enc_seqlen'],
#         model_dec_seqlen=config['data']['dec_seqlen'],
#         model_max_bars=config['data']['max_bars'],
#         pieces=pickle_load(config['data']['val_split']),
#         pad_to_same=True
#     )
#     print('[info]', '# training samples:', len(dset.pieces))

#     dloader = DataLoader(
#         dset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8)
#     dloader_val = DataLoader(
#         dset_val, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8)

#     return dset, dset_val, dloader, dloader_val


music_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
model_out_path = music_config['model']['output_path']
bs = music_config["batch_size"]
epochs = music_config['training']['max_epochs']


def _train():
    train_dset, val_dset, test_dset, train_dloader, val_dloader, test_dloader = get_dataloader(music_config)
    music_config.n_token = train_dset.vocab_size   # 333

    model = MusicCLIP(music_config, text_args)
    print(model.state_dict().keys())

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 5e-4)
    c_loss = ContrastiveLoss(bs)

    model.zero_grad()

    # model = model.double()
    start_time  = time.time()
    model.train()
    for epoch in range(epochs):
        print("Starting epoch ", epoch)
        for batch_idx, batch_samples in enumerate(train_dloader):
            model.zero_grad()
            batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)
            batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)
            batch_dec_tgt = batch_samples['dec_target'].permute(1, 0).to(device)
            batch_inp_bar_pos = batch_samples['bar_pos'].to(device)
            batch_inp_lens = batch_samples['length']
            batch_padding_mask = batch_samples['enc_padding_mask'].to(device)
            batch_rfreq_cls = batch_samples['rhymfreq_cls'].permute(1, 0).to(device)
            batch_polyph_cls = batch_samples['polyph_cls'].permute(1, 0).to(device)
            text = batch_samples['text_label'].permute(1, 0).to(device)
            y = batch_samples['pos'].permute(1, 0).to(device)

            lang_feats, vision_feats, pooled_output = model(
                text,
                batch_enc_inp, 
                batch_dec_inp, 
                batch_inp_bar_pos, 
                batch_rfreq_cls=None, 
                batch_polyph_cls=None, 
                batch_padding_mask=None,
                music_attention_mask = None,
            )
            loss = c_loss(pooled_output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("loss", loss.item())
            #training accuracy

            # save model 
            torch.save(model.state_dict(), model_out_path + "epoch{}_bs{}.pt".format(
                epoch = epochs, bs = bs,
            ))


def _inf():
    trained_mdl = ()
    # mdl = MusicCLIPInfer(saved_musicCLIP)
    # generate

    return



def train():
    mode = music_config["training"]["mode"]
    if mode == "TRAIN":
        _train()
    elif mode == "INFERENCE":
        _inf()
    else:
        raise ValueError("Unrecognized mode!")