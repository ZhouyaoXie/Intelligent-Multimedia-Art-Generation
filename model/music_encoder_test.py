from music_encoder import MusicEncoder
import sys, os, time, yaml 
import torch 
from musemorphose.utils import pickle_load
from musemorphose.dataloader import REMIFullSongTransformerDataset
from torch.utils.data import DataLoader

config_path = "config/default.yaml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

global trained_steps

device = config['training']['device']
trained_steps = config['training']['trained_steps']
lr_decay_steps = config['training']['lr_decay_steps']
lr_warmup_steps = config['training']['lr_warmup_steps']
no_kl_steps = config['training']['no_kl_steps']
kl_cycle_steps = config['training']['kl_cycle_steps']
kl_max_beta = config['training']['kl_max_beta']
free_bit_lambda = config['training']['free_bit_lambda']
max_lr, min_lr = config['training']['max_lr'], config['training']['min_lr']

ckpt_dir = config['training']['ckpt_dir']
params_dir = os.path.join(ckpt_dir, 'params/')
optim_dir = os.path.join(ckpt_dir, 'optim/')
pretrained_params_path = config['model']['pretrained_params_path']
pretrained_optim_path = config['model']['pretrained_optim_path']
ckpt_interval = config['training']['ckpt_interval']
log_interval = config['training']['log_interval']
val_interval = config['training']['val_interval']
constant_kl = config['training']['constant_kl']

recons_loss_ema = 0.
kl_loss_ema = 0.
kl_raw_ema = 0.

# for testing purpose just hardcode this value 
vocab_size = 333

if __name__ == "__main__":
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
  print ('[info]', '# training samples:', len(dset.pieces))

  dloader = DataLoader(dset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8)
  dloader_val = DataLoader(dset_val, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8)

  mconf = config['model']
  model = MusicEncoder(
    mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
    mconf['d_latent'], mconf['d_embed'], vocab_size,
    d_polyph_emb=mconf['d_polyph_emb'], d_rfreq_emb=mconf['d_rfreq_emb'],
    cond_mode=mconf['cond_mode']
  ).to(device)
  if pretrained_params_path:
    model.load_state_dict(torch.load(pretrained_params_path), strict = False)

    print("save encoder weights...")
    torch.save(model.state_dict(), "music_encoder_weight.pt")

  # number of data points to perform forward pass on in testing 
  test_count = 0
  num_to_test = 5 

  model.eval()
  for batch_idx, batch_samples in enumerate(dloader):
    model.zero_grad()
    batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)
    batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)
    batch_dec_tgt = batch_samples['dec_target'].permute(1, 0).to(device)
    batch_inp_bar_pos = batch_samples['bar_pos'].to(device)
    batch_inp_lens = batch_samples['length']
    batch_padding_mask = batch_samples['enc_padding_mask'].to(device)
    batch_rfreq_cls = batch_samples['rhymfreq_cls'].permute(1, 0).to(device)
    batch_polyph_cls = batch_samples['polyph_cls'].permute(1, 0).to(device)

    trained_steps += 1

    mu, logvar, dec_logits = model(
      batch_enc_inp, batch_dec_inp, 
      batch_inp_bar_pos, batch_rfreq_cls, batch_polyph_cls,
      padding_mask=batch_padding_mask
    )

    test_count += 1
    if test_count >= num_to_test:
      break 

