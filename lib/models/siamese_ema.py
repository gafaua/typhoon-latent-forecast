import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEMA(nn.Module):
    def __init__(self, base_encoder, dim=1024, out_dim=512, momentum=0.999) -> None:
        super(SiameseEMA, self).__init__()
        self.momentum = momentum

        self.encoder_q = nn.Sequential(
            base_encoder,
            nn.Linear(out_dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),   # hidden layer
            nn.Linear(dim, out_dim), # output layer
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),   # hidden layer
            nn.Linear(out_dim, out_dim) # output layer
            )

        self.encoder_k = copy.deepcopy(self.encoder_q)


    def forward(self, x1, x2):
        q = self.encoder_q(x1)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update()
            k = self.encoder_k(x2)
            k = F.normalize(k, dim=1)

        return q, k


    def _momentum_update(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)


def infoNCELoss(q, k, queue, T):
    # Einstein sum is more intuitive
    # positive logits: Nx1
    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
    # negative logits: NxK
    l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])
    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)

    # apply temperature
    logits /= T

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

    return F.cross_entropy(logits, labels)
