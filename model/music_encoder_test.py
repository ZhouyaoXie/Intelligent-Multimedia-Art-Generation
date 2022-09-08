import sys
import os
import time
import yaml
import torch
from torch.utils.data import DataLoader

# import module from the same directory
from music_encoder import MusicEncoder

print("importing module...")
# import module from parent directory
appended_path = None
print(sys.path)
for path in sys.path:
    for delimiter in ["\\", "/"]:
        if "Intelligent-Multimedia-Art-Generation{d}model".format(
            d=delimiter
        ) in path:
            appended_path = '{path}{d}..'.format(
                path=path,
                d=delimiter,
            )
            sys.path.append(appended_path)
            break
from musemorphose.dataloader import REMIFullSongTransformerDataset
from musemorphose.utils import pickle_load
if appended_path is not None:
    sys.path.remove(appended_path)

print("loading config...")
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


def test_dataloader():
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

    return dloader, dloader_val


def test_load_model():
    mconf = config['model']
    model = MusicEncoder(
        mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
        mconf['d_latent'], mconf['d_embed'], vocab_size,
        d_polyph_emb=mconf['d_polyph_emb'], d_rfreq_emb=mconf['d_rfreq_emb'],
        cond_mode=mconf['cond_mode']
    ).to(device)
    if pretrained_params_path:
        model.load_state_dict(torch.load(pretrained_params_path), strict=False)
        print("save encoder weights...")
        torch.save(model.state_dict(), "music_encoder_weight.pt")

    return model


def test_forward(dloader, num_to_test):
    test_count = 0
    model.eval()
    for batch_idx, batch_samples in enumerate(dloader):
        model.zero_grad()
        batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)
        batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)
        batch_dec_tgt = batch_samples['dec_target'].permute(1, 0).to(device)
        batch_inp_bar_pos = batch_samples['bar_pos'].to(device)
        batch_inp_lens = batch_samples['length']
        batch_padding_mask = batch_samples['enc_padding_mask'].to(device)
        batch_rfreq_cls = batch_samples['rhymfreq_cls'].permute(
            1, 0).to(device)
        batch_polyph_cls = batch_samples['polyph_cls'].permute(1, 0).to(device)

        trained_steps += 1

        mu, logvar, dec_logits = model(
            batch_enc_inp, batch_dec_inp,
            batch_inp_bar_pos, batch_rfreq_cls, batch_polyph_cls,
            padding_mask=batch_padding_mask
        )
        assert mu is not None and logvar is not None and dec_logits is not None
        print("batch_idx: {batch_idx}\nmu: {mu}\nlogvar: {logvar}\ndec_logits: {logits}".format(
            batch_idx=batch_idx,
            logvar=logvar,
            logits=dec_logits,
        ))

        test_count += 1
        if test_count >= num_to_test:
            break


if __name__ == "__main__":

    # test dataloader
    print("testing dataloader...")
    dloader, dloader_val = test_dataloader()

    print("testing MusicEncoder...")
    # test loading MusicEncoder
    model = test_load_model()

    # test forward
    print("testing music encoder forward...")
    test_forward(dloader, 3)
