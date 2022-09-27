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

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature = 1, alpha = 0.5, reduction = 'mean'):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reduction = reduction 

    def forward(self, x1, y1, y2=None, norm = True):
        """ Computes contrastive loss as the cross entropy over cosine similarities between
        paired inputs, with positive pairs having label 1 and negative pairs label 0
        https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py

        Params:
            x1: input embedding, shape [bs, embed_dim]
            y1: positive pair embedding, shape [bs, embed_dim]
            y2: [Optional] negative pair embedding, shape [bs, embed_dim]
                if negative pair is None, then use all other positive pairs as negative
            norm: normalize input embeddings or not, default to be True

        Returns:
            Contrastive loss
        """
        if norm:
            x1, y1, y2 = self.normalize(x1, y1, y2)
        
        if y2 is not None:
            # Explicit negative keys

            # Cosine between positive pairs
            positive_logit = torch.sum(x1 * y1, dim=1, keepdim=True)

            # Cosine between all x1-negative combinations
            negative_logits = x1 @ self.transpose(y2)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=x1.device)
        else:
            # Negative keys are implicitly off-diagonal positive keys.

            # Cosine between all combinations
            logits = x1 @ torch.transpose(y1)

            # Positive keys are the entries on the diagonal
            labels = torch.arange(len(x1), device=x1.device)

        return F.cross_entropy(logits / self.temperature, labels, reduction=self.reduction)

    def transpose(self, x):
        return x.transpose(-2, -1)

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]


# test 
if __name__ == "__main__":
    loss = ContrastiveLoss(batch_size = 3)

    x1 = torch.Tensor([[0, 0, 0.8, 0.7, 0],
                       [0.7, 0.9, 0, 0, 0],
                       [0.7, 0.9, 0, 0, 0]])
    y1 = torch.Tensor([[0.1, 0.2, 1, 1, 0.1],
                        [0.8, 0.7, 0, 0, 0],
                        [0.8, 0.7, 0, 0, 0]])
    y2 = torch.Tensor([[0.9, 0.8, 0.1, 0, 0.9],
                        [0.1, 0.1, 0.9, 0.8, 0.9],
                        [0.1, 0, 0.9, 1, 1],])
    print('this loss should be small: ', loss(x1, y1, y2))

    x1 = torch.Tensor([[0, 0, 0.8, 0.7, 0],
                       [0.7, 0.9, 0, 0, 0],
                       [0.7, 0.9, 0, 0, 0]])
    y1 = torch.Tensor([[0.9, 1, 0, 0.1, 1],
                        [0.1, 0.2, 0.5, 0.6, 0.5],
                        [0, 0.1, 0.9, 0.8, 0.7]])
    y2 = torch.Tensor([[0.1, 0.2, 0.9, 0.9, 0.3],
                        [0.6, 0.7, 0.2, 0.1, 0.2],
                        [0.8, 0.7, 0.2, 0.1, 0.2]])
    print('this loss should be big: ', loss(x1, y1, y2))