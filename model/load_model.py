import torch
import yaml

from clip import clip
from musemorphose.model.musemorphose import MuseMorphose

config_path = "config/default.yaml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
ckpt_path = "https://zenodo.org/record/5119525/files/musemorphose_pretrained_weights.pt?download=1"
device = config['training']['device']
t_con, m_con = config['text'], config['music']

def load_clip():
    clip_model = clip.load()
    return clip_model

def load_musemorphose():
    model = MuseMorphose(
        m_con['enc_n_layer'],
        m_con['enc_n_head'],
        m_con['enc_d_model'],
        m_con['enc_d_ff'],
        m_con['dec_n_layer'], 
        m_con['dec_n_head'], 
        m_con['dec_d_model'], 
        m_con['dec_d_ff'],
        m_con['d_latent'],
        m_con['d_embed'],
        m_con['vocab_size'],
        d_polyph_emb = 32,
        d_rfreq_emb = 32,
        cond_mode = m_con['cond_mode']
    ).to(device)
    model.eval()
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return model

def load():
    pass