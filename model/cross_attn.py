
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
