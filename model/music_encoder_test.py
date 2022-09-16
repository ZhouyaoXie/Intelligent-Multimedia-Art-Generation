import sys
import os
import time
import yaml
import torch
from torch.utils.data import DataLoader

# import module from the same directory
from music_transformer import MusicEncoder, MusicConfig

print("importing module...")
# import module from parent directory
appended_path = ""
print(sys.path)
delimiter = "/" if "/" in sys.path[0] else "\\"
for path in sys.path:
    if "Intelligent-Multimedia-Art-Generation{d}model".format(
        d=delimiter
    ) in path:
        appended_path = '{path}{d}..'.format(
            path=path,
            d=delimiter
        )
        break
sys.path.append(appended_path)
from musemorphose.dataloader import REMIFullSongTransformerDataset
from musemorphose.utils import pickle_load
sys.path.remove(appended_path)

print("loading config...")
config_path = "config/default.yaml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

global trained_steps

config = MusicConfig()

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

    return dset, dset_val, dloader, dloader_val


def test_load_model(dset):
    mconf = config['model']
    model = MusicEncoder(
        mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
        mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
        mconf['d_latent'], mconf['d_embed'], dset.vocab_size,
        d_polyph_emb=mconf['d_polyph_emb'], d_rfreq_emb=mconf['d_rfreq_emb'],
        cond_mode=mconf['cond_mode']
    ).to(config.device)
    if config.pretrained_params_path:
        model.load_state_dict(torch.load(config.pretrained_params_path), strict=False)
        print("save encoder weights...")
        torch.save(model.state_dict(), "music_encoder_weight.pt")

    return model


def test_forward(dloader, num_to_test):
    device = config.device
    global trained_steps 

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
        print("""batch_idx: {batch_idx}\n
            mu: {mu}\n
            logvar: {logvar}\n
            dec_logits: {logits}
        """.format(
            batch_idx=batch_idx,
            mu=mu[:,0],
            logvar=logvar[:,0],
            logits=dec_logits[:,:,0]
        ))

        test_count += 1
        if test_count >= num_to_test:
            break


if __name__ == "__main__":

    # test dataloader
    print("testing dataloader...")
    dset, dset_val, dloader, dloader_val = test_dataloader()

    print("testing MusicEncoder...")
    # test loading MusicEncoder
    model = test_load_model(dset)

    # test forward
    print("testing music encoder forward...")
    test_forward(dloader, 3)
