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
    def __init__(self, device, batch_size, temperature, alpha = 0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
        """
        logprobs = F.log_softmax(logits, dim = 1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        # print('logprobs ', logprobs)
        # print('multiply:', target * logprobs)
        # print('sum:', -(target * logprobs).sum())
        return loss

    def forward(self, x1, x2, flag, norm = True, include_negatives = True):
        """ Compute contrastive loss between two embeddings given flag +1/-1
        From https://github.com/edreisMD/ConVIRT-pytorch

        Args:
            x1, x2: two embedding tensors with the same shape, [bs, embed_dim]
            flag: 1 represents positive pairs, -1 represent negative pairs, size [bs]
            norm: boolean, whether to normalize input embeddings or not
        Returns:
            if flag is positive, return a small value if x1 and x2 are similar 
                and a large value if they are far away;
            if flag is negative, do the opposite
        """
        # Get (normalized) hidden1 and hidden2.
        if norm:
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
            
        batch_size = x1.shape[0]

        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        labels = labels.to(self.device)
        mask = torch.where(flag == 1, 1, -1)
        
        logits_ab = torch.matmul(x1, torch.transpose(x2, 0, 1)) / self.temperature
        logits_ba = torch.matmul(x2, torch.transpose(x1, 0, 1)) / self.temperature

        print(logits_ab)
        print('soft xent:', self.softXEnt(labels, logits_ab))

        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)

        return self.alpha * loss_a + (1 - self.alpha) * loss_b

# test 
if __name__ == "__main__":
    loss = ContrastiveLoss("cpu", 2, 1)

    x1 = torch.Tensor([[0, 0, 0.8, 0.7, 0],
                       [0.7, 0.9, 0, 0, 0]])
    x2 = torch.Tensor([[0.9, 0.8, 0.1, 0, 0.9],
                        [0.8, 0.7, 0, 0, 0]])
    flag = torch.Tensor([-1, 1])
    print('this loss should be small: ', loss(x1, x2, flag))

    # x1 = torch.Tensor([[0.1, 0.2, 0.8, 0.7, 0.9],
    #                    [0.7, 0.9, 0.2, 0.3, 0.1]])
    # x2 = torch.Tensor([[0.9, 0.8, 0.3, 0.2, 0.3],
    #                     [0.8, 0.7, 0.3, 0.1, 0.0]])
    # flag = torch.Tensor([1, -1])
    # print('this loss should be large: ', loss(x1, x2, flag))