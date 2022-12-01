import torch
from torch import nn
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from .music_transformer import VAETransformerEncoder 
from .music_encoder_utils import (
    TokenEmbedding, PositionalEncoding, weights_init
)
from .text_encoder import BertLayer , BertEmbeddings, BertPooler, BertPreTrainedModel, BertLayerNorm, BertPreTrained
from .cross_attn import MusicClIPXLayer
from .tokenization import BertTokenizer
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MusicCLIP(torch.nn.Module):
    def __init__(
        self,
        music_config,
        text_config,
    ):
        super().__init__()
        self.music_config = music_config 
        self.config = text_config
        self.music_seq_len = music_config['data']['enc_seqlen']
        self.text_seq_len = 128

        # initialzie music encoder
        self._init_music_encoder_from_config(music_config)

        # initialize cross attention layers
        # self.num_x_layers = music_config['x_attention']['num_x_layers']
        self.num_x_layers = text_config.xlayers
        self.x_layers = nn.ModuleList(
            [MusicClIPXLayer(text_config) for _ in range(self.num_x_layers)]
        )

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        self.max_seq_length = 128

        #Initialize text encoder
        self.embeddings = BertEmbeddings(text_config)
        self.pooler = BertPooler(text_config)
        self._init_bert_from_config(text_config)
        self.apply(self.init_bert_weights)


        #Initialize text encoder from pre-trained Weights
        self.bert = BertPreTrained.from_pretrained(
            "bert-base-uncased"
        )


        print("Printing the pre-trained Bert weights")
        from pprint import pprint
        pprint(vars(self.bert))




    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _init_music_encoder_from_config(self, config):
        self.token_emb = TokenEmbedding(config['data']['n_token'], config['model']['d_embed'], config['model']['enc_d_model'])
        self.pe = PositionalEncoding(config['model']['d_embed'])
        # self.dec_out_proj = nn.Linear(config['model']['dec_d_model'], config['data']['n_token'])
        self.encoder = VAETransformerEncoder(
            config['model']['enc_n_layer'], 
            config['model']['enc_n_head'], 
            config['model']['enc_d_model'], 
            config['model']['enc_d_ff'], 
            config['model']['d_latent'], 
            config['model'].get('enc_dropout', 0.1), 
            config['model'].get('enc_activation', 'relu')
        )

        self.emb_dropout = nn.Dropout(config['model'].get('enc_dropout', 0.1))

        if config['model']['pretrained_params_path'] is not None:
            self.load_state_dict(torch.load(config['model']['pretrained_params_path']), strict=False)
        else:
            weights_init(self)
        
        # add extra layer to project output from 512 to 768 dim
        self.out_proj = nn.Linear(config['model']['d_embed'], self.config.hidden_size)

    def _init_bert_from_config(self, config):
        # code from https://github.com/huggingface/transformers/blob/ad11b79e95acb3c89f994c725594ec52bd181fbf/src/transformers/models/bert/modeling_bert.py#L556
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.llayers)])
        self.gradient_checkpointing = False


    def encode_music(self, 
        enc_inp, 
        dec_inp, 
        # dec_inp_bar_pos, 
        # rfreq_cls=None, 
        # polyph_cls=None, 
        padding_mask=None,
        # use_attr_cls=True,
        batch_first = True
    ):
        # [shape of enc_inp] (seqlen_per_bar, bsize, n_bars_per_sample)
        enc_bt_size, enc_n_bars = enc_inp.size(1), enc_inp.size(2)
        # print('enc_inp', enc_inp.size()) # (128, 4, 512)
        enc_token_emb = self.token_emb(enc_inp)
        # print('enc_token_emb', enc_token_emb.size()) # (128, 4, 16, 512) [seq_len, bs, n_bars_per_sample, d_embed]

        # [shape of dec_inp] (seqlen_per_sample, bsize)
        # [shape of rfreq_cls & polyph_cls] same as above 
        # -- (should copy each bar's label to all corresponding indices)
        dec_token_emb = self.token_emb(dec_inp)

        enc_token_emb = enc_token_emb.reshape(
            enc_inp.size(0), -1, enc_token_emb.size(-1)
        )

        # print('enc_token_emb', enc_token_emb.size()) # (128, 64, 512) [seq_len, bs * n_bars_per_sample, d_embed]
        enc_inp = self.emb_dropout(enc_token_emb) + self.pe(enc_inp.size(0))
        dec_inp = self.emb_dropout(dec_token_emb) + self.pe(dec_inp.size(0))
        # print('enc_inp', enc_inp.size()) # (128, 64, 512)

        # [shape of padding_mask] (bsize, n_bars_per_sample, seqlen_per_bar)
        # -- should be `True` for padded indices (i.e., those >= seqlen of the bar), `False` otherwise
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))

        music_feats , music_feats_hidden, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)

        music_feats = music_feats.reshape(enc_inp.size(0), enc_bt_size, enc_n_bars, enc_token_emb.size(-1))
        music_feats = music_feats.mean(2)
        if batch_first:
            music_feats = music_feats.transpose(0, 1)
        # print('music_feats size returned by encode_music', music_feats.size())

        return music_feats


    def encode_text(
        self,
        sents,
        token_type_ids = None,
        attention_mask = None,
    ):
        train_features = convert_sents_to_features(
            sents, self.max_seq_length, self.tokenizer
        )

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).to(device)
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).to(device)
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).to(device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # pad to music seq_len (128)
        attention_mask = F.pad(
            input=attention_mask, 
            pad = (0, self.music_seq_len - self.text_seq_len, 0, 0)
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        lang_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        lang_feats = self.embeddings(input_ids, segment_ids)

        return lang_feats, lang_attention_mask


    def forward(
        self,
        sents,
        enc_inp, 
        dec_inp, 
        dec_inp_bar_pos, 
        num_bars,
        rfreq_cls=None, 
        polyph_cls=None, 
        padding_mask=None,
        music_attention_mask = None,
        token_type_ids = None,
    ):
        """ Adapted from https://github.com/airsplay/lxmert/blob/master/src/lxrt/modeling.py#L546
        
        """
        # Run music embedding layer
        music_feats = self.encode_music(
            enc_inp, 
            dec_inp, 
            # dec_inp_bar_pos, 
            # rfreq_cls, 
            # polyph_cls,
            padding_mask
        )
        # project music feats from 512 to 768
        music_feats = self.out_proj(music_feats)
        
        try:
            assert music_feats.size()[0] == 4
            assert music_feats.size()[1] == 128
            assert music_feats.size()[2] == 768
        except Exception:
            print('WARNING music_feats dimension mismatch, expect {s}, get {t}'.format(
                s = (4, 128, 768),
                t = music_feats.size(),
                )
            )

        # encode text
        lang_feats, lang_attention_mask = self.encode_text(
            sents,
            token_type_ids
        )
        lang_feats = F.pad(
            input=lang_feats, 
            pad = (0, 0, 0, self.music_seq_len - self.text_seq_len, 0, 0)
        )
        # Run language layers
        # for layer_module in self.layer:
        #     lang_feats = layer_module(lang_feats, lang_attention_mask)

        #Run language layers from pretrianed bert
        lang_feats = self.bert(lang_feats, lang_attention_mask)
        print("Shape of lang feats is ", lang_feats.shape)



        # print('lang_attention_mask size', lang_attention_mask.size()) # [4, 1, 1, 128]

        # # repeat lang_feats & lang_attention_mask for all music bars
        # lang_feats = lang_feats.repeat((num_bars, 1, 1))
        # lang_attention_mask = lang_attention_mask.repeat((num_bars, 1, 1, 1))
        # pad lang_feats seq_len to match with music encoder 
        music_seq_len = music_feats.size()[1]
        text_seq_len = lang_feats.size()[1]
        # print('music & text seq_len', music_seq_len, text_seq_len)
        
        try:
            assert lang_feats.size()[0] == 4
            assert lang_feats.size()[1] == 128
            assert lang_feats.size()[2] == 768
        except Exception as e:
            print('WARNING music_feats dimension mismatch, expect {s}, get {t}'.format(
                s = (4, 128, 768),
                t = music_feats.size(),
                )
            )

        # Run cross-modality layers
        for layer_module in self.x_layers:
            lang_feats, music_feats = layer_module(lang_feats, lang_attention_mask,
                                                  music_feats, music_attention_mask)

        #pooled output to run the contrasitve loss from the hidden token of the first token in final layer
        pooled_output = self.pooler(lang_feats)

        return lang_feats, music_feats , pooled_output, lang_attention_mask


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

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