from copy import deepcopy
import time
import numpy as np
import torch
from torch import nn
import os, sys
from typing import Optional, Tuple
import yaml 
from scipy.stats import entropy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from .music_transformer import VAETransformerEncoder, VAETransformerDecoder 
from .music_encoder_utils import (
    TokenEmbedding, PositionalEncoding, weights_init
)
from .text_encoder import BertLayer , BertEmbeddings, BertPooler, BertPreTrainedModel
from .cross_attn import MusicClIPXLayer
from .utils import nucleus, pickle_load, numpy_to_tensor, temperatured_softmax, tensor_to_numpy, get_beat_idx, word2event
from .remi2midi import remi2midi


config_path = "config/default.yaml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
vocab_path = config['data']['vocab_path']
use_attr_cls = config['model']['use_attr_cls']

vocab = pickle_load(vocab_path)[0]
idx2event = pickle_load(vocab_path)[1]
vocab_size = len(vocab)
event2idx = vocab

out_dir = config['generate']['out_dir']

class MusicCLIPInfer(BertPreTrainedModel):
    def __init__(
        self,
        model,
        music_config,
        text_config,
    ):
        super().__init__()
        self.model = model
        self.music_config = music_config 
        self.config = text_config

        # cross attention layers are in the encoder object that is passed in
        # self.num_x_layers = music_config.num_x_layers
        # self.x_layers = nn.ModuleList(
        #     [MusicClIPXLayer(text_config) for _ in range(self.num_x_layers)]
        # )

        self._init_music_decoder_from_config(music_config)


    def _init_music_decoder_from_config(self, config):
        # self.token_emb = TokenEmbedding(config.n_token, config.d_embed, config.enc_d_model)
        # self.pe = PositionalEncoding(config.d_embed)
        self.dec_out_proj = nn.Linear(config.dec_d_model, config.n_token)
        # self.encoder = VAETransformerEncoder(
        #     config.enc_n_layer, 
        #     config.enc_n_head, 
        #     config.enc_d_model, 
        #     config.enc_d_ff, 
        #     config.d_latent, 
        #     config.enc_dropout, 
        #     config.enc_activation
        # )

        if config.use_attr_cls:
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

        if config.use_attr_cls:
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


    def decode_music(
        self, 
        dec_inp, dec_inp_bar_pos, rfreq_cls, polyph_cls,
        enc_bt_size = config['data']['enc_seqlen'], 
        enc_n_bars = config['data']['max_bars'], 
    ):
        #decoder part
        mu = 0
        logvar = 1
        vae_latent = self.reparameterize(mu, logvar)
        # enc_bt_size & enc_n_bars come from: enc_inp.size(1), enc_inp.size(2)
        # enc_inp is 'enc_input' in dataset
        # its length should be self.model_enc_seqlen
        vae_latent_reshaped = vae_latent.reshape(enc_bt_size, enc_n_bars, -1)

        # [shape of dec_inp] (seqlen_per_sample, bsize)
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

        return dec_out, dec_logits, mu, logvar


    def text_encoder(self, lang_feats, lang_attention_mask):
        for layer_module in self.layer:
            lang_feats = layer_module(lang_feats, lang_attention_mask)
        return lang_feats, lang_attention_mask


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

        # music = get_sample_latent_generator(mu_0, logvar_0)
        # # enc_inp, 
        # #     dec_inp, 
        # #     dec_inp_bar_pos, 
        # #     rfreq_cls, 
        # #     polyph_cls,
        # #     padding_mask

        music_feats, dec_logits, mu, logvar = self.decode_music(
            dec_inp, dec_inp_bar_pos, rfreq_cls, polyph_cls
        )

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
 

    # below are methods for music decoder generation 
    def reparameterize(self, mu, logvar, use_sampling=True, sampling_var=1.):
        std = torch.exp(0.5 * logvar).to(mu.device)
        if use_sampling:
            eps = torch.randn_like(std).to(mu.device) * sampling_var
        else:
            eps = torch.zeros_like(std).to(mu.device)

        return eps * std + mu

    def get_sampled_latent(self, inp, padding_mask=None, use_sampling=False, sampling_var=0.):
        token_emb = self.model.token_emb(inp)
        enc_inp = self.model.emb_dropout(token_emb) + self.model.pe(inp.size(0))

        _, mu, logvar = self.model.encoder(enc_inp, padding_mask=padding_mask)
        mu, logvar = mu.reshape(-1, mu.size(-1)), logvar.reshape(-1, mu.size(-1))
        vae_latent = self.reparameterize(mu, logvar, use_sampling=use_sampling, sampling_var=sampling_var)

        return vae_latent

    def get_sampled_latent_inference(self, sampling_var=0.):
        mu = 0
        logvar =1 
        mu, logvar = mu.reshape(-1, mu.size(-1)), logvar.reshape(-1, mu.size(-1))
        vae_latent = self.reparameterize(mu, logvar, sampling_var=sampling_var)

        return vae_latent

    def generate_on_latent_ctrl_vanilla_truncate(self, 
        latents, event2idx, idx2event, 
        rfreq_cls = None, polyph_cls = None, 
        max_events=12800, primer=None,
        max_input_len=1280, truncate_len=512, 
        nucleus_p=0.9, temperature=1.2
      ):
        latent_placeholder = torch.zeros(max_events, 1, latents.size(-1)).to(device)
        rfreq_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
        polyph_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
        print ('[info] rhythm cls: {} | polyph_cls: {}'.format(rfreq_cls, polyph_cls))

        if primer is None:
            generated = [event2idx['Bar_None']]
        else:
            generated = [event2idx[e] for e in primer]
            latent_placeholder[:len(generated), 0, :] = latents[0].squeeze(0)
            rfreq_placeholder[:len(generated), 0] = rfreq_cls[0]
            polyph_placeholder[:len(generated), 0] = polyph_cls[0]
            
        target_bars, generated_bars = latents.size(0), 0

        steps = 0
        time_st = time.time()
        cur_pos = 0
        failed_cnt = 0

        cur_input_len = len(generated)
        generated_final = deepcopy(generated)
        entropies = []

        while generated_bars < target_bars:
            if len(generated) == 1:
                dec_input = numpy_to_tensor([generated], device=device).long()
            else:
                dec_input = numpy_to_tensor([generated], device=device).permute(1, 0).long()

            latent_placeholder[len(generated)-1, 0, :] = latents[ generated_bars ]
            rfreq_placeholder[len(generated)-1, 0] = rfreq_cls[ generated_bars ]
            polyph_placeholder[len(generated)-1, 0] = polyph_cls[ generated_bars ]

            dec_seg_emb = latent_placeholder[:len(generated), :]
            dec_rfreq_cls = rfreq_placeholder[:len(generated), :]
            dec_polyph_cls = polyph_placeholder[:len(generated), :]

            # sampling
            with torch.no_grad():
                logits = self.generate(dec_input, dec_seg_emb, dec_rfreq_cls, dec_polyph_cls)
                logits = tensor_to_numpy(logits[0])
                probs = temperatured_softmax(logits, temperature)
                word = nucleus(probs, nucleus_p)
                word_event = idx2event[word]

            if 'Beat' in word_event:
                event_pos = get_beat_idx(word_event)
            if not event_pos >= cur_pos:
                failed_cnt += 1
                print ('[info] position not increasing, failed cnt:', failed_cnt)
                if failed_cnt >= 128:
                    print ('[FATAL] model stuck, exiting ...')
                    return generated
                continue
            else:
                cur_pos = event_pos
                failed_cnt = 0

            if 'Bar' in word_event:
                generated_bars += 1
                cur_pos = 0
                print ('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated_final)))
            if word_event == 'PAD_None':
                continue

            if len(generated) > max_events or (word_event == 'EOS_None' and generated_bars == target_bars - 1):
                generated_bars += 1
                generated.append(event2idx['Bar_None'])
                print ('[info] gotten eos')
                break

            generated.append(word)
            generated_final.append(word)
            entropies.append(entropy(probs))

            cur_input_len += 1
            steps += 1

            assert cur_input_len == len(generated)
            if cur_input_len == max_input_len:
                generated = generated[-truncate_len:]
                latent_placeholder[:len(generated)-1, 0, :] = latent_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0, :]
                rfreq_placeholder[:len(generated)-1, 0] = rfreq_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]
                polyph_placeholder[:len(generated)-1, 0] = polyph_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]

                print ('[info] reset context length: cur_len: {}, accumulated_len: {}, truncate_range: {} ~ {}'.format(
                    cur_input_len, len(generated_final), cur_input_len-truncate_len, cur_input_len-1
                ))
                cur_input_len = len(generated)

        assert generated_bars == target_bars
        print ('-- generated events:', len(generated_final))
        print ('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))
        return generated_final[:-1], time.time() - time_st, np.array(entropies)


    def generate(self, inp, dec_seg_emb, rfreq_cls=None, polyph_cls=None, keep_last_only=True):
        token_emb = self.model.token_emb(inp)
        dec_inp = self.model.emb_dropout(token_emb) + self.model.pe(inp.size(0))

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


    def generate_music(self, n_pieces = 1, rfreq_cls=None, polyph_cls=None, keep_last_only=True):
        times = []
        piece_entropies = []
        for p in range(n_pieces):
            out_file = os.path.join(out_dir, 'id{}_poly{}_rhym{}'.format(
                p, "None" if rfreq_cls is None else rfreq_cls, "None" if polyph_cls is None else polyph_cls
            ))      
            print ('[info] writing to ...', out_file)
            if os.path.exists(out_file + '.txt'):
                print ('[info] file exists, skipping ...')
                continue

            p_latents = self.get_sampled_latent_inference()
            song, t_sec, entropies = self.generate_on_latent_ctrl_vanilla_truncate(
                                        p_latents, event2idx, idx2event,
                                        p_rfreq_cls = None, p_polyph_cls = None, 
                                        max_input_len=config['generate']['max_input_dec_seqlen'], 
                                        truncate_len=min(512, config['generate']['max_input_dec_seqlen'] - 32), 
                                        nucleus_p=config['generate']['nucleus_p'], 
                                        temperature=config['generate']['temperature']
                                    )
            times.append(t_sec)

            song = word2event(song, idx2event)
            print (*song, sep='\n', file=open(out_file + '.txt', 'a'))
            remi2midi(song, out_file + '.mid', enforce_tempo=True)

            # save metadata of the generation
            # np.save(out_file + '-POLYCLS.npy', tensor_to_numpy(p_polyph_cls))
            # np.save(out_file + '-RHYMCLS.npy', tensor_to_numpy(p_rfreq_cls))
            print ('[info] piece entropy: {:.4f} (+/- {:.4f})'.format(
                entropies.mean(), entropies.std()
            ))
            piece_entropies.append(entropies.mean())

        print ('[time stats] {} songs, generation time: {:.2f} secs (+/- {:.2f})'.format(
            n_pieces, np.mean(times), np.std(times)
        ))
        print ('[entropy] {:.4f} (+/- {:.4f})'.format(
            np.mean(piece_entropies), np.std(piece_entropies)
        ))