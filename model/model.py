import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from clip.model import Transformer, LayerNorm

class MusicCLIP:
    def __init__(self,
                 embed_dim: int,
                 # vision
                 n_layer: int, 
                 n_head: int, 
                 d_model: int, 
                 d_ff: int, 
                 d_vae_latent: int, 
                 dropout: float, 
                 activation: str,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):

        # text 
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers

        # vision
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model 
        self.d_ff = d_ff
        self.d_vae_latent = d_vae_latent
        self.dropout = dropout 
        self.activation = activation

        # text
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # music
        self.tr_encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_head, d_ff, dropout, activation
        )
        self.tr_encoder = nn.TransformerEncoder(
            self.tr_encoder_layer, n_layer
        )
        self.fc_mu = nn.Linear(d_model, d_vae_latent)
        self.fc_logvar = nn.Linear(d_model, d_vae_latent)

    def initialize_params(self):
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

        # music
        