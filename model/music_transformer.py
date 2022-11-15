import torch
from torch import nn
from model.music_encoder_utils import generate_causal_mask
import yaml 
import os 


# """ Class for reading music transformer config from yaml file """
# class MusicConfig:
#     def __init__(self, config_path = None):
#         if config_path is None:
#             config_path = "config/default.yaml"
#         config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

#         # training parameters
#         self.device = config['training']['device']
#         self.trained_steps = config['training']['trained_steps']
#         self.lr_decay_steps = config['training']['lr_decay_steps']
#         self.lr_warmup_steps = config['training']['lr_warmup_steps']
#         self.no_kl_steps = config['training']['no_kl_steps']
#         self.kl_cycle_steps = config['training']['kl_cycle_steps']
#         self.kl_max_beta = config['training']['kl_max_beta']
#         self.free_bit_lambda = config['training']['free_bit_lambda']
#         self.max_lr, self.min_lr = config['training']['max_lr'], config['training']['min_lr']
#         self.mode = config['training']['mode']

#         self.ckpt_dir = config['training']['ckpt_dir']
#         self.params_dir = os.path.join(self.ckpt_dir, 'params/')
#         self.optim_dir = os.path.join(self.ckpt_dir, 'optim/')

#         # music transformer 
#         mconf = config['model']
#         self.pretrained_params_path = mconf['pretrained_params_path']
#         self.pretrained_optim_path = mconf['pretrained_optim_path']
#         self.d_polyph_emb = mconf['d_polyph_emb']
#         self.d_rfreq_emb = mconf['d_rfreq_emb']
#         self.cond_mode = mconf['cond_mode']
#         self.n_rfreq_cls = mconf.get('n_rfreq_cls', 8)
#         self.n_polyph_cls = mconf.get('n_polyph_cls', 8)
#         self.use_attr_cls = mconf['use_attr_cls']

#         # music encoder 
#         self.enc_n_layer = mconf['enc_n_layer']
#         self.enc_n_head = mconf['enc_n_head']
#         self.enc_d_model = mconf['enc_d_model']
#         self.enc_d_ff = mconf['enc_d_ff']
#         self.enc_dropout = mconf.get('enc_dropout', 0.1)
#         self.enc_activation = mconf.get('enc_activation', 'relu')


#         # music decoder 
#         self.dec_n_layer = mconf['dec_n_layer']
#         self.dec_n_head = mconf['dec_n_head']
#         self.dec_d_model = mconf['dec_d_model']
#         self.dec_d_ff = mconf['dec_d_ff']
#         self.d_latent = mconf['d_latent']
#         self.d_embed = mconf['d_embed']
#         self.dec_dropout = mconf.get('dec_dropout', 0.1)
#         self.dec_activation = mconf.get('dec_activation', 'relu')

#         # cross_attention
#         self.num_x_layers = config['x_attention']['num_x_layers']

#         # training
#         self.ckpt_interval = config['training']['ckpt_interval']
#         self.log_interval = config['training']['log_interval']
#         self.val_interval = config['training']['val_interval']
#         self.constant_kl = config['training']['constant_kl']

#         self.config = config 


""" Class for a transformer encoder """
class VAETransformerEncoder(nn.Module):
  def __init__(self, 
    n_layer, n_head, d_model, d_ff, 
    d_vae_latent, dropout=0.1, activation='relu'
  ):
    """ Initialize a transformer encoder.
    
    Params:
      n_layer: the number of sub-encoder-layers in the encoder
      n_head: 
    """
    super(VAETransformerEncoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_vae_latent = d_vae_latent
    self.enc_dropout = dropout
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

    return out, hidden_out, mu, logvar


class VAETransformerDecoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, d_seg_emb, dropout=0.1, activation='relu', cond_mode='in-attn'):
    super(VAETransformerDecoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_seg_emb = d_seg_emb
    self.dec_dropout = dropout
    self.activation = activation
    self.cond_mode = cond_mode

    if cond_mode == 'in-attn':
      self.seg_emb_proj = nn.Linear(d_seg_emb, d_model, bias=False)
    elif cond_mode == 'pre-attn':
      self.seg_emb_proj = nn.Linear(d_seg_emb + d_model, d_model, bias=False)

    self.decoder_layers = nn.ModuleList()
    for i in range(n_layer):
      self.decoder_layers.append(
        nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
      )

  def forward(self, x, seg_emb):
    if not hasattr(self, 'cond_mode'):
      self.cond_mode = 'in-attn'
    attn_mask = generate_causal_mask(x.size(0)).to(x.device)
    # print (attn_mask.size())

    if self.cond_mode == 'in-attn':
      seg_emb = self.seg_emb_proj(seg_emb)
    elif self.cond_mode == 'pre-attn':
      x = torch.cat([x, seg_emb], dim=-1)
      x = self.seg_emb_proj(x)

    out = x
    for i in range(self.n_layer):
      if self.cond_mode == 'in-attn':
        print("\n\n In Music Transformer, \n")
        print("shape of out is ", out.shape)
        print("shape of seg_emb is ", seg_emb.shape)
        out += seg_emb
      out = self.decoder_layers[i](out, src_mask=attn_mask)

    return out