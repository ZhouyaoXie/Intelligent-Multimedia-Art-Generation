import sys
import yaml
import torch
from torch.utils.data import DataLoader

# import module from the same directory
from model.music_transformer import MusicConfig
from model.model import MusicCLIP
from dataloader.dataloader_updated import get_dataloader

# print("importing module...")
# # import module from parent directory
# appended_path = ""
# print(sys.path)
# delimiter = "/" if "/" in sys.path[0] else "\\"
# for path in sys.path:
#     if "Intelligent-Multimedia-Art-Generation{d}model".format(
#         d=delimiter
#     ) in path:
#         appended_path = '{path}{d}..'.format(
#             path=path,
#             d=delimiter
#         )
#         break
# sys.path.append(appended_path)
# from musemorphose.dataloader import REMIFullSongTransformerDataset
# from musemorphose.utils import pickle_load
# sys.path.remove(appended_path)

print("loading config...")
config_path = "config/default.yaml"

global trained_steps

config = MusicConfig(config_path)

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
    data_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    train_dset, val_dset, test_dset, train_dloader, val_dloader, test_dloader = test_dataloader(data_config)

    # need to get from true dataloader 
    config.n_token = 333

    model = MusicCLIP(config)
    # print(model.state_dict().keys())
