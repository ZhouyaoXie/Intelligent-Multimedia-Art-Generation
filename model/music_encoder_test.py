from music_encoder import MusicEncoder
import sys, os, time, yaml 
import torch 

config_path = "config/default.yaml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

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

# for testing purpose
vocab_size = 1024

if __name__ == "__main__":
  mconf = config['model']
  model = MusicEncoder(
    mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
    mconf['d_latent'], mconf['d_embed'], vocab_size,
    d_polyph_emb=mconf['d_polyph_emb'], d_rfreq_emb=mconf['d_rfreq_emb'],
    cond_mode=mconf['cond_mode']
  ).to(device)
  if pretrained_params_path:
    model.load_state_dict(torch.load(pretrained_params_path), strict = False)

    print("save encoder and decoder weights...")
    torch.save(model.state_dict(), "music_encoder_weight.pt")