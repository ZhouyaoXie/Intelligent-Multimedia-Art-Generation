import sys, os, random, time
sys.path.append('./model')

from dataloader import REMIFullSongTransformerDataset
from model.musemorphose import MuseMorphose
from model.transformer_encoder import VAETransformerEncoder
from utils import pickle_load

import torch
import yaml
import numpy as np

print("reading config file...")
config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
data_dir = config['data']['data_dir']
vocab_path = config['data']['vocab_path']
data_split = 'pickles/test_pieces.pkl'
ckpt_path = sys.argv[2]


if __name__ == "__main__":
  dset = REMIFullSongTransformerDataset(
    data_dir, vocab_path, 
    do_augment=False,
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['generate']['dec_seqlen'],
    model_max_bars=config['generate']['max_bars'],
    pieces=pickle_load(data_split),
    pad_to_same=False
  )
  
  mconf = config['model']
  model = MuseMorphose(
    mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
    mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
    mconf['d_latent'], mconf['d_embed'], dset.vocab_size,
    d_polyph_emb=mconf['d_polyph_emb'], d_rfreq_emb=mconf['d_rfreq_emb'],
    cond_mode=mconf['cond_mode']
  ).to(device)
  model.eval()
  print("load MuseMorphose weights...")
  model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

  # save encoder & decoder weights
  print("save encoder and decoder weights...")
  torch.save(model.encoder.state_dict(), "encoder_weights.pt")
  torch.save(model.decoder.state_dict(), "decoder_weights.pt")

  # test load encoder 
  print("test loading encoder")
  encoder = VAETransformerEncoder(mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], 
                                        mconf['enc_d_ff'], mconf['d_latent'], 0.1, 'relu')
  encoder.load_state_dict(torch.load("encoder_weights.pt"))
  encoder.eval()
  print("TEST PASSED")