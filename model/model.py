import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from clip.model import Transformer, LayerNorm
from musemorphose.model.transformer_encoder import VAETransformerEncoder
from musemorphose.model.transformer_helpers import (
  weights_init, PositionalEncoding, TokenEmbedding
)

class MusicCLIP:
    def __init__(self,
                 embed_dim: int,
                 # music
                 n_layer: int, 
                 n_head: int, 
                 d_model: int, 
                 d_ff: int, 
                 d_vae_latent: int, 
                 dropout: float, 
                 activation: str,
                 n_token: int,
                 d_embed: int,
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

        # music
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model 
        self.d_ff = d_ff
        self.d_vae_latent = d_vae_latent
        self.dropout = dropout 
        self.activation = activation
        self.n_token = n_token
        self.d_embed = d_embed

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
        self.token_emb = TokenEmbedding(n_token, d_embed, d_model)
        self.pe = PositionalEncoding(d_embed)
        self.encoder = VAETransformerEncoder(
        n_layer, n_head, d_model, d_ff, d_vae_latent, dropout, activation
        )
        self.emb_dropout = nn.Dropout(self.dropout)

        # music attributes
        self.d_rfreq_emb = 32
        self.d_polyph_emb = 32
        self.rfreq_attr_emb = TokenEmbedding(8, 32, 32)
        self.polyph_attr_emb = TokenEmbedding(8, 32, 32)

        # initialize parameters
        self.initialize_params()

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

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x