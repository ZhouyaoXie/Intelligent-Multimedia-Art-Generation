import time
import random
import numpy as np
import yaml 
import os, sys
import pickle
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

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

music_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
model_out_path = music_config['training']['output_path']
bs = music_config["data"]["batch_size"]
epochs = music_config['training']['max_epochs']
lr = music_config['training']['max_lr']
# max number of epochs to train the decoder on during inference
MAX_INFERENCE_EPOCH = music_config['training']['max_inference_epoch']
# stop decoder training if the contrastive loss is smaller than this value
MAX_INFERENCE_LOSS = music_config['training']['max_inference_loss']

def save_loss(losses, epoch):
    with open(f"{model_out_path}epoch{epoch}_losses", "wb") as f:
        pickle.dump(losses, f)

def _train(music_config, text_args):
    train_dset, val_dset, test_dset, train_dloader, val_dloader, test_dloader = get_dataloader(music_config)
    if 'n_token' not in music_config['data'] or music_config['data']['n_token'] is None:
        music_config['data']['n_token'] = train_dset.vocab_size   # 333 in musemorphose; 404 in our data

    model = MusicCLIP(music_config, text_args)
    print(model.state_dict().keys())

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 5e-4)
    c_loss = ContrastiveLoss(bs)

    model.zero_grad()

    start_time  = time.time()
    model.train()
    for epoch in range(epochs):
        print("Starting epoch ", epoch)
        losses = []
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
            music_pooled = model.music_pool(music_feats)
            loss = c_loss(music_pooled, pooled_output, y)  # music_pooled & pooled_output should have shape (bs, emd_dim)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 1000 == 0:
                losses.append(loss.item())
                print("Current batch:", batch_idx, "loss:", loss.item())
            #training accuracy
        
        save_loss(losses, epoch)
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
    # get input params for inference
    train_dset, _, _, train_dloader, _, _ = get_dataloader(music_config)
    # train_dset = torch.tensor(train_dset)
    dec_inp = torch.tensor(train_dset[0]['dec_input']).reshape(-1,1).permute(1,0).to(device)
    dec_inp_bar_pos = torch.tensor(train_dset[0]['bar_pos']).reshape(-1,1).to(device)
    rfreq_cls = torch.tensor(train_dset[0]['rhymfreq_cls']).reshape(-1,1).permute(1,0).to(device)
    polyph_cls = torch.tensor(train_dset[0]['polyph_cls']).reshape(-1,1).permute(1,0).to(device)
    
    # dec_inp = np.array([1])
    # dec_inp_bar_pos = np.array([1])


    # load saved MusicCLIP model
    # model = None
    model = MusicCLIP(music_config, text_args)
    # if model_save_path is None:
    #     model_save_path = model_out_path + "epoch{epoch}_bs{bs}_lr{lr}.pt".format(
    #         epoch = epochs, bs = bs, lr = lr,
    #     )
    # model.load_state_dict(torch.load(model_save_path))
    # model.eval()

    print("\n\n\nStarting the inference pipeline\n\n\n")

    # initiate MusicCLIPInfer model
    infer_model = MusicCLIPInfer(model, music_config, text_args)

    # initialize training optimizer and loss
    optimizer = optim.Adam(infer_model.parameters(), lr=lr, weight_decay = 5e-4)
    c_loss = ContrastiveLoss(bs)
    c_loss = torch.nn.CrossEntropyLoss()

    start_time  = time.time()
    infer_model.train()
    rfreq_cls = None
    polyph_cls =None
    for epoch in range(MAX_INFERENCE_EPOCH):
        print("Starting epoch ", epoch)
        lang_feats, music_feats, pooled_output = infer_model(
            text,
            dec_inp, 
            dec_inp_bar_pos, 
            rfreq_cls, 
            polyph_cls
        )
        print("shaped of the pooled output shape  is ", pooled_output.shape)
        y = torch.tensor(np.ones(pooled_output.shape[0]))
        # y = y.type(torch.LongTensor)
        loss = c_loss(pooled_output.reshape(-1), y)
        # loss = c_loss(lang_feats, music_feats, POSITIVE)

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
        _inf("the test case", music_config, text_config)
    else:
        raise ValueError("Unrecognized mode!")

# def __main__():
#     train(music_config, text_args)
# train(music_config, text_args)
