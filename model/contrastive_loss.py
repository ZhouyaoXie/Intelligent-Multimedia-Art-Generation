"""
References: 
- CLIP paper refers to three literature:
  - Sohn, multi-class n-pair loss: 
      https://papers.nips.cc/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf
  - Zhang et al., first to apply multi-class n-pair loss to multimodal context and make it asymmetric:
      https://arxiv.org/pdf/2010.00747.pdf
      Code: https://github.com/edreisMD/ConVIRT-pytorch/blob/master/loss/nt_xent.py
- Supervised contrastive learning, a work that proposes a generalized version of n-pair loss (SupConLoss):
    https://arxiv.org/pdf/2004.11362.pdf
- pytorch implementation of SupConLoss:
    https://github.com/HobbitLong/SupContrast
- a blog post on contrastive, triplet, and n-pair losses:
    https://kobiso.github.io/research/research-n-pair-loss/
- pytorch implementation of n-pair loss:
    https://github.com/ChaofWang/Npair_loss_pytorch/blob/master/Npair_loss.py
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from matplotlib import pyplot as plt 

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature = 1, alpha = 0.5, reduction = 'mean'):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reduction = reduction 

    def forward(self, query, embed, y, norm = True, inference = False):
        """ Computes contrastive loss as the cross entropy over cosine similarities between
        paired inputs, with positive pairs having label 1 and negative pairs label 0
        https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py

        Params:
            query: input embedding, shape [bs, embed_dim]
            embed: pooled text embedding to compare query with, shape [bs, embed_dim]
            y: label, shape [bs]. Should be [1, 0, 1, 0, ...]
            norm: normalize input embeddings or not, default to be True

        Returns:
            Contrastive loss
        """
        if not inference:
            # input will always be (positive, negative, positive, negative, ...) in pairs
            # process input first 
            y1, y2 = embed[::2, :], embed[1::2, :]
            x1 = query[::2, :]
            return self._loss(x1, y1, y2, norm)
            # print('loss: ', loss)
            return loss
        else:
            return self._loss(query, embed, None, norm)
        
    def _loss(self, x1, y1, y2, norm):
        if norm:
            x1, y1, y2 = self.normalize(x1, y1, y2)

        # print('x1 size', x1.size())    # (bs, emb_dim)
        N, D = x1.shape[0], x1.shape[1]
        if y2:
            y2 = y2.reshape((N, 1, D))  # (bs, num_neg_examples, emb_dim)
            # print('y1 y2 size', y1.size(), y2.size())

        # Cosine between positive pairs
        positive_logit = torch.sum(x1 * y1, dim=1, keepdim=True)
        # print('positive logits shape: ', positive_logit.shape)  # should be (bs, 1)

        if y2:
            # print('x1 shape, y2 shape: ', x1.shape, y2.shape)
            x1 = x1.unsqueeze(1)
            # print('x1 shape: ', x1.shape)  # should be (bs, 1, emb_dim) @ (bs, emb_dim, num_neg_examples)
            # print('y2 trans shape: ', y2.shape)

            negative_logits = x1 @ self.transpose(y2)
            # print('negative logits shape: ', negative_logits.shape) # should be (bs, 1, num_neg_examples)

            negative_logits = negative_logits.squeeze(1)  # should be (bs, num_neg_examples)
            # print('negative logits shape: ', negative_logits.shape)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=x1.device)

            return F.cross_entropy(logits / self.temperature, labels, reduction=self.reduction)

        else:
            logits = positive_logit.float()
            labels = torch.ones(logits.shape, dtype = torch.long, device = x1.device).float()

            return F.binary_cross_entropy(logits / self.temperature, labels, reduction=self.reduction)


    def transpose(self, x):
        return x.transpose(-2, -1)

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]


# test 
if __name__ == "__main__":
    bs, seq_len, emb_dim = 4, 128, 786
    loss = ContrastiveLoss(batch_size = bs)
    random_loss = []
    for _ in range(100):
        music_pooled = torch.rand((bs, emb_dim), dtype=torch.float64)
        text_pooled = torch.rand((bs, emb_dim), dtype=torch.float64)
        random_label = torch.randint(0, 2, (bs,))
        random_loss.append(loss(music_pooled, text_pooled, random_label, norm = True))

    print(random_loss)

    plt.hist(random_loss)
    plt.show()