from model import MusicCLIP
from load_model import load_clip
import sys, yaml

config_path = "config/default.yaml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
t_con, m_con = config['text'], config['music']

mdl = MusicCLIP(embed_dim = t_con['embed_dim'],
                 # music
                 n_layer = m_con['enc_n_layer'],
                 n_head = m_con['enc_n_head'],
                 d_model = m_con['enc_d_model'],
                 d_ff = m_con['enc_d_ff'],
                 d_vae_latent = m_con['d_latent'],
                 dropout = 0.1,
                 activation = "relu",
                 n_token = 1024, # should be dset.vocab_size
                 d_embed = m_con['d_embed'],
                 # text
                 context_length = t_con['context_length'],
                 vocab_size = t_con['vocab_size'],
                 transformer_width = t_con['transformer_width'],
                 transformer_heads = t_con['transformer_heads'],
                 transformer_layers = t_con['transformer_layers'])

mdl = load_clip()