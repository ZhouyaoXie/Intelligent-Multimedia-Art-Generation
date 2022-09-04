import torch
from torch import nn
import torch.nn.functional as F
from music_encoder_utils import TokenEmbedding, PositionalEncoding, weights_init

class VAETransformerEncoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, d_vae_latent, dropout=0.1, activation='relu'):
    super(VAETransformerEncoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_vae_latent = d_vae_latent
    self.dropout = dropout
    self.activation = activation

    self.tr_encoder_layer = nn.TransformerEncoderLayer(
      d_model, n_head, d_ff, dropout, activation
    )
    self.tr_encoder = nn.TransformerEncoder(
      self.tr_encoder_layer, n_layer
    )

    self.fc_mu = nn.Linear(d_model, d_vae_latent)
    self.fc_logvar = nn.Linear(d_model, d_vae_latent)

  def forward(self, x, padding_mask=None):
    out = self.tr_encoder(x, src_key_padding_mask=padding_mask)
    hidden_out = out[0, :, :]
    mu, logvar = self.fc_mu(hidden_out), self.fc_logvar(hidden_out)

    return hidden_out, mu, logvar


class MusicEncoder(nn.Module):
  def __init__(self, enc_n_layer, enc_n_head, enc_d_model, enc_d_ff,
    d_vae_latent, d_embed, n_token,
    enc_dropout=0.1, enc_activation='relu',
    d_rfreq_emb=32, d_polyph_emb=32,
    n_rfreq_cls=8, n_polyph_cls=8,
    is_training=True, use_attr_cls=True,
    cond_mode='in-attn'
  ):
    self.enc_n_layer = enc_n_layer
    self.enc_n_head = enc_n_head
    self.enc_d_model = enc_d_model
    self.enc_d_ff = enc_d_ff
    self.enc_dropout = enc_dropout
    self.enc_activation = enc_activation

    self.d_vae_latent = d_vae_latent
    self.n_token = n_token
    self.is_training = is_training

    self.cond_mode = cond_mode
    self.token_emb = TokenEmbedding(n_token, d_embed, enc_d_model)
    self.d_embed = d_embed
    self.pe = PositionalEncoding(d_embed)
    self.encoder = VAETransformerEncoder(
      enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, d_vae_latent, enc_dropout, enc_activation
    )

    if use_attr_cls:
      self.d_rfreq_emb = d_rfreq_emb
      self.d_polyph_emb = d_polyph_emb
      self.rfreq_attr_emb = TokenEmbedding(n_rfreq_cls, d_rfreq_emb, d_rfreq_emb)
      self.polyph_attr_emb = TokenEmbedding(n_polyph_cls, d_polyph_emb, d_polyph_emb)
    else:
      self.rfreq_attr_emb = None
      self.polyph_attr_emb = None

    self.emb_dropout = nn.Dropout(self.enc_dropout)
    self.apply(weights_init)
