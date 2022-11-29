import os
import time
import random
import numpy as np
import yaml 

import torch
import torch.optim as optim

from .model import MusicCLIP, convert_sents_to_features

from dataloader.dataloader_updated import get_dataloader

from .inference import MusicCLIPInfer
from config.text_config import text_args
from model.inference import MusicCLIPInfer
from .contrastive_loss import ContrastiveLoss

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("the set device is ", device)

DEBUG = True 
if DEBUG:
    print("=" * 30 + "\nWARNING: DEBUG MODE\n" + "=" * 30)

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

# contrastive loss if music and text match 
POSITIVE = 1
music_config_global = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
trained_steps = music_config_global['training']['trained_steps']

# music_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
# model_out_path = music_config['training']['output_path']
# bs = music_config["data"]["batch_size"]
# epochs = music_config['training']['max_epochs']
# lr = music_config['training']['max_lr']
# # max number of epochs to train the decoder on during inference
# MAX_INFERENCE_EPOCH = music_config['training']['max_inference_epoch']
# # stop decoder training if the contrastive loss is smaller than this value
# MAX_INFERENCE_LOSS = music_config['training']['max_inference_loss']



def _train(music_config, text_args):
    model_out_path = music_config['training']['output_path']
    bs = music_config["data"]["batch_size"]
    epochs = music_config['training']['max_epochs']
    lr = music_config['training']['max_lr']
    min_lr = music_config['training']['min_lr']
    lr_decay_steps = music_config['training']['lr_decay_steps']
    lr_warmup_steps = music_config['training']['lr_warmup_steps']
    
    if not os.path.exists(model_out_path):
        os.makedirs(model_out_path)
        
    train_dset, val_dset, test_dset, train_dloader, val_dloader, test_dloader = get_dataloader(music_config)
    if 'n_token' not in music_config['data'] or music_config['data']['n_token'] is None:
        music_config['data']['n_token'] = train_dset.vocab_size   # 333 in musemorphose; 404 in our data

    model = MusicCLIP(music_config, text_args)
    print(model.state_dict().keys())

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
      optimizer, lr_decay_steps, eta_min=min_lr
    )  # added (same as musemorphose)

    c_loss = ContrastiveLoss(bs)
    if DEBUG:
        c_loss = torch.nn.CrossEntropyLoss()

    model.zero_grad()

    start_time  = time.time()
    model.train()
    for epoch in range(epochs):
        print("Starting epoch ", epoch)
        for batch_idx, batch_samples in tqdm(enumerate(train_dloader)):
            if batch_idx > 0.1 * len(train_dloader):
                break
            model.zero_grad()
            batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)
            batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)
            batch_inp_bar_pos = batch_samples['bar_pos'].to(device)
            batch_padding_mask = batch_samples['enc_padding_mask'].to(device)
            batch_rfreq_cls = batch_samples['rhymfreq_cls'].permute(1, 0).to(device)
            batch_polyph_cls = batch_samples['polyph_cls'].permute(1, 0).to(device)
            # text = torch.Tensor(batch_samples['text_label']).permute(1, 0).to(device)
            text = batch_samples['text_label']
            y = batch_samples['pos'].float().to(device)
            
            global trained_steps
            trained_steps += 1
            
            lang_feats, music_feats, pooled_output, lang_attention_mask = model(
                text,
                batch_enc_inp, 
                batch_dec_inp, 
                batch_inp_bar_pos, 
                music_config['data']['max_bars'],
                rfreq_cls=batch_rfreq_cls, 
                polyph_cls=batch_polyph_cls, 
                padding_mask=batch_padding_mask,
                music_attention_mask = None,
            )

            if not DEBUG:
                loss = c_loss(music_feats, lang_feats, y)
            else:
                pooled_output =  model.pooled_proj(pooled_output)
                try:
                    loss = c_loss(pooled_output.reshape(-1), y)
                except Exception as e:
                    print(str(e))
                    print("Shape of pooled_output: ", pooled_output.shape, "; shape of y: ", y.shape)

            # anneal learning rate, added
            if trained_steps < lr_warmup_steps:
                curr_lr = lr * trained_steps / lr_warmup_steps
                optimizer.param_groups[0]['lr'] = curr_lr
            else:
                scheduler.step(trained_steps - lr_warmup_steps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#             print("loss", loss.item())
            # training accuracy
            if batch_idx % 1000 == 0:
                print("batch_idx:", batch_idx, "loss:", loss.item())

        # save model checkpoint after every epoch
        torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, 
            model_out_path + 'model_epoch{epoch}_bs{bs}_lr{lr}_ckpt_epoch{curr_epoch}.pth'.format(
                epoch = epochs, bs = bs, lr = lr, curr_epoch = epoch
            )
        )

    # save model 
    torch.save(model.state_dict(), model_out_path + "epoch{epoch}_bs{bs}_lr{lr}.pt".format(
        epoch = epochs, bs = bs, lr = lr,
    ))


def _inf(text, music_config, text_args, model_save_path = None, n_pieces = 1):
    model_out_path = music_config['training']['output_path']
    bs = music_config["data"]["batch_size"]
    epochs = music_config['training']['max_epochs']
    lr = music_config['training']['max_lr']
    # max number of epochs to train the decoder on during inference
    MAX_INFERENCE_EPOCH = music_config['training']['max_inference_epoch']
    # stop decoder training if the contrastive loss is smaller than this value
    MAX_INFERENCE_LOSS = music_config['training']['max_inference_loss']
    
    # get input params for inference
    train_dset, _, _, train_dloader, _, _ = get_dataloader(music_config)
    dec_inp = train_dset[0]['dec_input'].permute(1, 0).to(device)
    dec_inp_bar_pos = train_dset[0]['bar_pos'].to(device)
    rfreq_cls = train_dset[0]['rhymfreq_cls'].permute(1, 0).to(device)
    polyph_cls = train_dset[0]['polyph_cls'].permute(1, 0).to(device)
    
    # load saved MusicCLIP model
    model = MusicCLIP(music_config, text_args)
    if model_save_path is None:
        model_save_path = model_out_path + "epoch{epoch}_bs{bs}_lr{lr}.pt".format(
            epoch = epochs, bs = bs, lr = lr,
        )
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # initiate MusicCLIPInfer model
    infer_model = MusicCLIPInfer(model, music_config, text_args)

    # initialize training optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 5e-4)

    c_loss = ContrastiveLoss(bs)

    start_time  = time.time()
    infer_model.train()
    for epoch in range(MAX_INFERENCE_EPOCH):
        print("Starting epoch ", epoch)
        lang_feats, music_feats, pooled_output = infer_model(
            text,
            dec_inp, 
            dec_inp_bar_pos, 
            rfreq_cls, 
            polyph_cls
        )
        loss = c_loss(pooled_output, POSITIVE)
        print("loss", loss.item())
        if loss < MAX_INFERENCE_LOSS:
            break 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    training_end_time = time.time()
    print("training ended after {e} epochs, took {t}s".format(
        e = epoch,
        t = round(training_end_time - start_time)
    ))

    # generate music using the updated decoder
    infer_model.generate_music(
        n_pieces,
        rfreq_cls,
        polyph_cls,
        keep_last_only = True
    )
    generate_end_time = time.time()
    print("generation ended, took {}s".format(round(generate_end_time - start_time)))


def train(music_config, text_config = text_args):
    mode = music_config["training"]["mode"]
    if mode == "TRAIN":
        _train(music_config, text_config)
    elif mode == "INFERENCE":
        _inf()
    else:
        raise ValueError("Unrecognized mode!")
