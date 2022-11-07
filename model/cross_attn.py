
from torch import nn

from .text_encoder import *

class MusicClIPXLayer(nn.Module):
	def __init__(self, config):
		super().__init__()
		# The cross-attention Layer
		self.visual_attention = BertCrossattLayer(config)

		# Self-attention Layers
		self.lang_self_att = BertSelfattLayer(config)
		self.visn_self_att = BertSelfattLayer(config)

		# Intermediate and Output Layers (FFNs)
		self.lang_inter = BertIntermediate(config)
		self.lang_output = BertOutput(config)
		self.visn_inter = BertIntermediate(config)
		self.visn_output = BertOutput(config)

	def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
		assert lang_input is not None 
		assert visn_input is not None 
		print('lang_input size', lang_input.size(), 'visn_input size', visn_input.size()) 
		# [4,20,768]; [128,64,512]
		# [bs, seq_len, d_latent], [seq_len, bs, d_latent]
		# 1. bert seq_len should be 128
		# 2. why music bs is 64?
		# 3. add 1 linear layer from 512 -> 768
		# Cross Attention
		lang_att_output = self.visual_attention(
			lang_input, visn_input, ctx_att_mask=visn_attention_mask)
		visn_att_output = self.visual_attention(
			visn_input, lang_input, ctx_att_mask=lang_attention_mask)
		return lang_att_output, visn_att_output

	def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
		# Self Attention
		lang_att_output = self.lang_self_att(
			lang_input, lang_attention_mask)
		visn_att_output = self.visn_self_att(
			visn_input, visn_attention_mask)
		return lang_att_output, visn_att_output

	def output_fc(self, lang_input, visn_input):
		# FC layers
		lang_inter_output = self.lang_inter(lang_input)
		visn_inter_output = self.visn_inter(visn_input)

		# Layer output
		lang_output = self.lang_output(lang_inter_output, lang_input)
		visn_output = self.visn_output(visn_inter_output, visn_input)
		return lang_output, visn_output

	def forward(self, lang_feats, lang_attention_mask,
				visn_feats, visn_attention_mask):
		lang_att_output = lang_feats
		visn_att_output = visn_feats

		lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask,
															visn_att_output, visn_attention_mask)
		lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask,
															visn_att_output, visn_attention_mask)
		lang_output, visn_output = self.output_fc(
			lang_att_output, visn_att_output)

		return lang_output, visn_output

class XLayer_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_x_layers = config.x_layers
        self.x_layers = nn.ModuleList(
            [MusicClIPXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, lang_feats, lang_attention_mask,
                music_feats, music_attention_mask=None):
        for layer_module in self.x_layers:
            lang_feats, visn_feats = layer_module(lang_feats, lang_attention_mask,
                                                  visn_feats, visn_attention_mask)

        return lang_feats, music_feats

if __name__ == "__main__":

    # test cross attention with text and text
    data_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    dset, dset_val, dloader, dloader_val = test_dataloader(data_config)
    model = MusicClIPXLayer(config)
    print(model.state_dict().keys())
