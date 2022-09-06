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
  def __init__(self, 
    enc_n_layer, enc_n_head, enc_d_model, enc_d_ff,
    d_vae_latent, d_embed, n_token,
    enc_dropout=0.1, enc_activation='relu',
    d_rfreq_emb=32, d_polyph_emb=32,
    n_rfreq_cls=8, n_polyph_cls=8,
    is_training=True, use_attr_cls=True,
    cond_mode='in-attn'
  ):
    super().__init__()
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


  def forward(self, enc_inp, dec_inp, dec_inp_bar_pos, rfreq_cls=None, polyph_cls=None, padding_mask=None):
    # [shape of enc_inp] (seqlen_per_bar, bsize, n_bars_per_sample)
    enc_bt_size, enc_n_bars = enc_inp.size(1), enc_inp.size(2)
    enc_token_emb = self.token_emb(enc_inp)

    # [shape of dec_inp] (seqlen_per_sample, bsize)
    # [shape of rfreq_cls & polyph_cls] same as above 
    # -- (should copy each bar's label to all corresponding indices)
    dec_token_emb = self.token_emb(dec_inp)

    enc_token_emb = enc_token_emb.reshape(
      enc_inp.size(0), -1, enc_token_emb.size(-1)
    )
    enc_inp = self.emb_dropout(enc_token_emb) + self.pe(enc_inp.size(0))
    dec_inp = self.emb_dropout(dec_token_emb) + self.pe(dec_inp.size(0))

    # [shape of padding_mask] (bsize, n_bars_per_sample, seqlen_per_bar)
    # -- should be `True` for padded indices (i.e., those >= seqlen of the bar), `False` otherwise
    if padding_mask is not None:
      padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))

    _, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
    vae_latent = self.reparameterize(mu, logvar)
    vae_latent_reshaped = vae_latent.reshape(enc_bt_size, enc_n_bars, -1)

    dec_seg_emb = torch.zeros(dec_inp.size(0), dec_inp.size(1), self.d_vae_latent).to(vae_latent.device)
    for n in range(dec_inp.size(1)):
      # [shape of dec_inp_bar_pos] (bsize, n_bars_per_sample + 1)
      # -- stores [[start idx of bar #1, sample #1, ..., start idx of bar #K, sample #1, seqlen of sample #1], [same for another sample], ...]
      for b, (st, ed) in enumerate(zip(dec_inp_bar_pos[n, :-1], dec_inp_bar_pos[n, 1:])):
        dec_seg_emb[st:ed, n, :] = vae_latent_reshaped[n, b, :]

    if rfreq_cls is not None and polyph_cls is not None and self.use_attr_cls:
      dec_rfreq_emb = self.rfreq_attr_emb(rfreq_cls)
      dec_polyph_emb = self.polyph_attr_emb(polyph_cls)
      dec_seg_emb_cat = torch.cat([dec_seg_emb, dec_rfreq_emb, dec_polyph_emb], dim=-1)
    else:
      dec_seg_emb_cat = dec_seg_emb

    dec_out = self.decoder(dec_inp, dec_seg_emb_cat)
    dec_logits = self.dec_out_proj(dec_out)

    return mu, logvar, dec_logits