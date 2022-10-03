import torch
import torch.nn as nn


class InceptionLoss(nn.Module):
    def __init__(self):
        super(InceptionLoss, self).__init__()
        self.out_loss = nn.CrossEntropyLoss()
        self.aux1_loss = nn.CrossEntropyLoss()
        self.aux2_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, aux_outputs1, aux_outputs2, targets):
        out_loss = self.out_loss(outputs, targets)
        aux1_loss = None
        if aux_outputs1 is not None:
            aux1_loss = self.aux1_loss(aux_outputs1, targets)
        aux2_loss = None
        if aux_outputs2 is not None:
            aux2_loss = self.aux2_loss(aux_outputs2, targets)
        total_loss = out_loss
        if aux_outputs1 is not None:
            total_loss = out_loss + 0.3 * aux1_loss + 0.3 * aux2_loss
        return out_loss
