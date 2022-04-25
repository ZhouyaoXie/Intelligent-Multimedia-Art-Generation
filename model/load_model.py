import torch
import yaml

from clip import clip
from model import MusicCLIP
from musemorphose.model.musemorphose import MuseMorphose

config_path = "config/default.yaml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
ckpt_path = "https://zenodo.org/record/5119525/files/musemorphose_pretrained_weights.pt?download=1"
device = config['training']['device']
t_con, m_con = config['text'], config['music']

def load_clip(name = "ViT-B/32"):
    clip_model = clip.load(name)
    if type(clip_model) == tuple: clip_model = clip_model[0]
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

def load(clip_name):
    mdl = MusicCLIP(m_con['d_embed'],
                 # music
                    m_con['enc_n_layer'],
                    m_con['enc_n_head'],
                    m_con['enc_d_model'],
                    m_con['enc_d_ff'],
                    m_con['d_latent'],
                    0.1,
                    "relu",
                    m_con['vocab_size'],
                    m_con['d_embed'],
                    # text
                    context_length = t_con['context_length'],
                    vocab_size = t_con['vocab_size'],
                    transformer_width = t_con['transformer_width'],
                    transformer_heads = t_con['transformer_heads'],
                    transformer_layers = t_con['transformer_layers'])

    # text
    clip_model = load_clip(clip_name)
    mdl.transformer = clip_model.transformer
    mdl.token_embedding = clip_model.token_embedding
    mdl.positional_embedding = clip_model.positional_embedding
    mdl.ln_final = clip_model.ln_final
    mdl.text_projection = clip_model.text_projection
    mdl.logit_scale = clip_model.logit_scale

    # music
    music_model = load_musemorphose()
    mdl.token_emb = music_model.token_emb
    mdl.pe = music_model.pe
    mdl.encoder = music_model.encoder
    mdl.emb_dropout = music_model.emb_dropout
    mdl.rfreq_attr_emb = music_model.rfreq_attr_emb
    mdl.polyph_attr_emb = music_model.polyph_attr_emb

    return mdl