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
from .text_encoder import BertLayer , BertEmbeddings, BertPooler, BertPreTrainedModel
from .cross_attn import MusicClIPXLayer
from .tokenization import BertTokenizer



class MusicCLIP(BertPreTrainedModel):
    def __init__(
        self,
        music_config,
        text_config,
    ):
        super().__init__()
        self.music_config = music_config 
        self.config = text_config

        # initialzie music encoder
        self._init_music_encoder_from_config(music_config)

        # initialize cross attention layers
        self.num_x_layers = music_config.num_x_layers
        self.x_layers = nn.ModuleList(
            [MusicClIPXLayer(text_config) for _ in range(self.num_x_layers)]
        )

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        self.max_seq_length = 20

        #Initialize text encoder
        self.embeddings = BertEmbeddings(text_config)
        self.pooler = BertPooler(text_config)
        self.apply(self.init_bert_weights)
        self._init_bert_from_config(text_config)


    def _init_music_transformer_from_config(self, config):
        self.token_emb = TokenEmbedding(config.n_token, config.d_embed, config.enc_d_model)
        self.pe = PositionalEncoding(config.d_embed)
        # self.dec_out_proj = nn.Linear(config.dec_d_model, config.n_token)
        self.encoder = VAETransformerEncoder(
            config.enc_n_layer, 
            config.enc_n_head, 
            config.enc_d_model, 
            config.enc_d_ff, 
            config.d_latent, 
            config.enc_dropout, 
            config.enc_activation
        )

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
        # dec_inp_bar_pos, 
        # rfreq_cls=None, 
        # polyph_cls=None, 
        padding_mask=None,
        # use_attr_cls=True,
    ):
        # [shape of enc_inp] (seqlen_per_bar, bsize, n_bars_per_sample)
        # enc_bt_size, enc_n_bars = enc_inp.size(1), enc_inp.size(2)
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

        music_feats , music_feats_hidden, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
        return music_feats


    # class text_encoder(BertPreTrainedModel):
    #     def __init__(self, configs):
    #         super().__init__()
    #         self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_bert_layers)])

    #     def forward(self, lang_feats, lang_attention_mask):
    #         for layer_module in self.layer:
    #             lang_feats = layer_module(lang_feats, lang_attention_mask)
    #         return lang_feats, lang_attention_mask

    # def text_encoder(self, lang_feats, lang_attention_mask):
    #     for layer_module in self.layer:
    #         lang_feats = layer_module(lang_feats, lang_attention_mask)
    #     return lang_feats, lang_attention_mask


    def forward(
        self,
        sents,
        enc_inp, 
        dec_inp, 
        dec_inp_bar_pos, 
        rfreq_cls=None, 
        polyph_cls=None, 
        padding_mask=None,
        music_attention_mask = None,
        token_type_ids = None
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


        train_features = convert_sents_to_features(
            sents, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()


        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        lang_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        lang_feats = self.embeddings(input_ids, segment_ids)
        # Run language layers
        for layer_module in self.layer:
            lang_feats, lang_attention_mask = layer_module(lang_feats, lang_attention_mask)

        # Run cross-modality layers
        for layer_module in self.x_layers:
            lang_feats, music_feats = layer_module(lang_feats, lang_attention_mask,
                                                  music_feats, music_attention_mask)

        #pooled output to run the contrasitve loss from the hidden token of the first token in final layer
        pooled_output = self.pooler(lang_feats)
        return lang_feats, music_feats , pooled_output


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids)
        )

    return features