import torch
from torch import nn
import os, sys
from typing import Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from .music_transformer import VAETransformerEncoder, VAETransformerDecoder 
from .music_encoder_utils import (
    TokenEmbedding, PositionalEncoding, weights_init
)
from .text_encoder import BertLayer, BertConfig , BertEmbeddings, BertAttention, BertAttOutput, BertCrossattLayer, BertSelfattLayer, BertIntermediate, BertOutput, BertPreTrainedModel
from .cross_attn import MusicClIPXLayer


class MusicCLIP(BertPreTrainedModel):
    def __init__(
        self,
        music_config,
        text_config = None,
    ):
        super().__init__()
        self.music_config = music_config 
        self.config = text_config

        if text_config is not None:
            self.n_cross_layers = text_config.num_x_layers
        self._init_music_transformer_from_config(music_config)

        # self._init_bert_from_config(text_config)
        # self.x_layers = nn.ModuleList(
        #     [MusicClIPXLayer(text_config) for _ in range(self.n_cross_layers)]
        # )


        self.num_x_layers = config.x_layers
        self.x_layers = nn.ModuleList(
            [MusicClIPXLayer(config) for _ in range(self.num_x_layers)]
        )

        # self.initialize_params()


        #Initialize text encoder
        self.embeddings = BertEmbeddings(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)


    def _init_music_transformer_from_config(self, config, use_attr_cls = True):
        self.token_emb = TokenEmbedding(config.n_token, config.d_embed, config.enc_d_model)
        self.pe = PositionalEncoding(config.d_embed)
        self.dec_out_proj = nn.Linear(config.dec_d_model, config.n_token)
        self.encoder = VAETransformerEncoder(
            config.enc_n_layer, 
            config.enc_n_head, 
            config.enc_d_model, 
            config.enc_d_ff, 
            config.d_latent, 
            config.enc_dropout, 
            config.enc_activation
        )

        if use_attr_cls:
            self.decoder = VAETransformerDecoder(
                config.dec_n_layer, 
                config.dec_n_head, 
                config.dec_d_model, 
                config.dec_d_ff, 
                config.d_latent + config.d_polyph_emb + config.d_rfreq_emb,
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
                config.d_latent,
                dropout = config.dec_dropout, 
                activation = config.dec_activation,
                cond_mode = config.cond_mode
            )

        if use_attr_cls:
            self.rfreq_attr_emb = TokenEmbedding(config.n_rfreq_cls, config.d_rfreq_emb, config.d_rfreq_emb)
            self.polyph_attr_emb = TokenEmbedding(config.n_polyph_cls, config.d_polyph_emb, config.d_polyph_emb)
        else:
            self.rfreq_attr_emb = None
            self.polyph_attr_emb = None

        self.emb_dropout = nn.Dropout(config.enc_dropout)

        if config.pretrained_params_path is not None:
            self.load_state_dict(torch.load(config.pretrained_params_path), strict=False)
        else:
            weights_init(self)

    def _init_bert_from_config(self, config):
        # code from https://github.com/huggingface/transformers/blob/ad11b79e95acb3c89f994c725594ec52bd181fbf/src/transformers/models/bert/modeling_bert.py#L556
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False


    def encode_music(self, 
        enc_inp, 
        dec_inp, 
        dec_inp_bar_pos, 
        rfreq_cls=None, 
        polyph_cls=None, 
        padding_mask=None,
        use_attr_cls=True,
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

        if rfreq_cls is not None and polyph_cls is not None and use_attr_cls:
            dec_rfreq_emb = self.rfreq_attr_emb(rfreq_cls)
            dec_polyph_emb = self.polyph_attr_emb(polyph_cls)
            dec_seg_emb_cat = torch.cat([dec_seg_emb, dec_rfreq_emb, dec_polyph_emb], dim=-1)
        else:
            dec_seg_emb_cat = dec_seg_emb

        dec_out = self.decoder(dec_inp, dec_seg_emb_cat)
        dec_logits = self.dec_out_proj(dec_out)

        return mu, logvar, dec_logits


    class text_encoder(BertPreTrainedModel):
        def __init__(self, config):
            super().__init__()
            self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_bert_layers)])

        def forward(self, lang_feats, lang_attention_mask):
            for layer_module in self.layer:
                lang_feats = layer_module(lang_feats, lang_attention_mask)
            return lang_feats, lang_attention_mask



    def encode_text(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    # logger.warning(
                    #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    # )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # TODO: do we need this class or no?
        # return BaseModelOutputWithPastAndCrossAttentions(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_decoder_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attentions,
        #     cross_attentions=all_cross_attentions,
        # )
        return 

    def forward(
        self,
        lang_feats,
        lang_attention_mask,
        enc_inp, 
        dec_inp, 
        dec_inp_bar_pos, 
        rfreq_cls=None, 
        polyph_cls=None, 
        padding_mask=None,
        music_attention_mask = None,
    ):
        """ Adapted from https://github.com/airsplay/lxmert/blob/master/src/lxrt/modeling.py#L546
        
        """
        # Run music embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.
        music_feats = self.encode_music(
            enc_inp, 
            dec_inp, 
            dec_inp_bar_pos, 
            rfreq_cls, 
            polyph_cls,
            padding_mask
        )

        # Run language layers
        for layer_module in self.layer:
            lang_feats = layer_module(lang_feats, lang_attention_mask)

        # Run relational layers
        # for layer_module in self.r_layers:
        #     visn_feats = layer_module(visn_feats, visn_attention_mask)

        # Run cross-modality layers
        for layer_module in self.x_layers:
            lang_feats, music_feats = layer_module(lang_feats, lang_attention_mask,
                                                  music_feats, music_attention_mask)

        return lang_feats, music_feats
 

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