import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from clip.model import Transformer, LayerNorm
from music_transformer import VAETransformerEncoder, VAETransformerDecoder 
from music_encoder_utils import (
    TokenEmbedding, PositionalEncoding, weights_init, generate_causal_mask
)


class MusicCLIP(nn.Module):
    def __init__(
        self,
        music_config,
        text_config
    ):
        super().__init__()
        self.music_config = music_config 
        self.text_config = text_config

        self._init_music_transformer_from_config(music_config)
        self._init_bert_from_config(text_config)

        self.initialize_params()


    def _init_music_transformer_from_config(self, config):
        self.token_emb = TokenEmbedding(config.n_token, config.d_embed, config.enc_d_model)
        self.pe = PositionalEncoding(config.d_embed)
        self.dec_out_proj = nn.Linear(config.dec_d_model, config.n_token)
        self.encoder = VAETransformerEncoder(
            config.enc_n_layer, 
            config.enc_n_head, 
            config.enc_d_model, 
            config.enc_d_ff, 
            config.d_vae_latent, 
            config.enc_dropout, 
            config.enc_activation
        )

        if config.use_attr_cls:
            self.decoder = VAETransformerDecoder(
                config.dec_n_layer, 
                config.dec_n_head, 
                config.dec_d_model, 
                config.dec_d_ff, 
                config.d_vae_latent + config.d_polyph_emb + config.d_rfreq_emb,
                dropout = config.dec_dropout, 
                activation = config.dec_activation,
                cond_mode = config.cond_mode
            )
        else:
            self.decoder = VAETransformerDecoder(
                config.dec_n_layer, 
                config.dec_n_head, 
                config.dec_d_model, 
                config.dec_d_ff, 
                config.d_vae_latent,
                dropout = config.dec_dropout, 
                activation = config.dec_activation,
                cond_mode = config.cond_mode
            )

        if config.use_attr_cls:
            self.rfreq_attr_emb = TokenEmbedding(config.n_rfreq_cls, config.d_rfreq_emb, config.d_rfreq_emb)
            self.polyph_attr_emb = TokenEmbedding(config.n_polyph_cls, config.d_polyph_emb, config.d_polyph_emb)
        else:
            self.rfreq_attr_emb = None
            self.polyph_attr_emb = None

        self.emb_dropout = nn.Dropout(self.enc_dropout)

    
    def _init_bert_from_config(self, config):
        # TODO: load BERT config and initialize text encoder 
        pass


    def initialize_params(self):
        # music
        weights_init(self)

        # text
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def encode_music(self, 
        enc_inp, 
        dec_inp, 
        dec_inp_bar_pos, 
        rfreq_cls=None, 
        polyph_cls=None, 
        padding_mask=None
    ):
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


    def encode_text(self):
        # TODO: BERT text encoder's forward method 
        pass 


    # below are methods for music decoder generation 
    def reparameterize(self, mu, logvar, use_sampling=True, sampling_var=1.):
        std = torch.exp(0.5 * logvar).to(mu.device)
        if use_sampling:
            eps = torch.randn_like(std).to(mu.device) * sampling_var
        else:
            eps = torch.zeros_like(std).to(mu.device)

        return eps * std + mu

    def get_sampled_latent(self, inp, padding_mask=None, use_sampling=False, sampling_var=0.):
        token_emb = self.token_emb(inp)
        enc_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))

        _, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
        mu, logvar = mu.reshape(-1, mu.size(-1)), logvar.reshape(-1, mu.size(-1))
        vae_latent = self.reparameterize(mu, logvar, use_sampling=use_sampling, sampling_var=sampling_var)

        return vae_latent

    def generate(self, inp, dec_seg_emb, rfreq_cls=None, polyph_cls=None, keep_last_only=True):
        token_emb = self.token_emb(inp)
        dec_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))

        if rfreq_cls is not None and polyph_cls is not None:
            dec_rfreq_emb = self.rfreq_attr_emb(rfreq_cls)
            dec_polyph_emb = self.polyph_attr_emb(polyph_cls)
            dec_seg_emb_cat = torch.cat([dec_seg_emb, dec_rfreq_emb, dec_polyph_emb], dim=-1)
        else:
            dec_seg_emb_cat = dec_seg_emb

        out = self.decoder(dec_inp, dec_seg_emb_cat)
        out = self.dec_out_proj(out)

        if keep_last_only:
            out = out[-1, ...]

        return out


    def compute_loss(self, mu, logvar, beta, fb_lambda, dec_logits, dec_tgt):
        recons_loss = F.cross_entropy(
        dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
        ignore_index=self.n_token - 1, reduction='mean'
        ).float()

        kl_raw = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(dim=0)
        kl_before_free_bits = kl_raw.mean()
        kl_after_free_bits = kl_raw.clamp(min=fb_lambda)
        kldiv_loss = kl_after_free_bits.mean()

        return {
            'beta': beta,
            'total_loss': recons_loss + beta * kldiv_loss,
            'kldiv_loss': kldiv_loss,
            'kldiv_raw': kl_before_free_bits,
            'recons_loss': recons_loss
        }