import torch
import torch.nn as nn

# Funciones de pérdida #


class mse_loss(nn.Module):
    def __init__(self, mode):
        """
            @param mode: 'mean' o 'sum'
        """
        super(mse_loss, self).__init__()
        self.mode = mode
        if self.mode == 'sum':
            self.mse = nn.MSELoss(reduction='sum')
        else:
            self.mse = nn.MSELoss(reduction='mean')

    def forward(self, prediction, target):
        loss = self.mse(prediction, target)
        if self.mode == 'sum':
            loss /= (len(prediction))
        return loss


class bce_loss(nn.Module):  # entropía cruzada binaria
    def __init__(self, pos_weight=None):
        super(bce_loss, self).__init__()
        # BCE con logits para AMP/autocast. Entradas deben ser logits crudos (sin sigmoid).
        # pos_weight ayuda con el desbalance de clases (99% fondo, 1% estrellas)
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction='mean', pos_weight=pos_weight)

    def forward(self, prediction, target):
        loss = self.bce_loss(prediction, target)
        return loss

