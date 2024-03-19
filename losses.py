import torch
import torch.nn.functional as F
from torch.nn import Module
import torch.nn as nn
from torch.autograd import Variable


class DiceLoss(Module):
    def __init__(self, weights):
        super(DiceLoss, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        pred = F.sigmoid(inputs)

        channels = inputs.size(1)

        score = 0

        for i in range(0, channels):
            pred_channel = pred[:, i, :, :]
            target_channel = targets[:, i, :, :]
            channel_weight = self.weights[i]

            pred_batch = pred_channel.view(pred_channel.size(
                0), pred_channel.size(1)*pred_channel.size(2))
            target_batch = target_channel.view(target_channel.size(
                0), target_channel.size(1)*target_channel.size(2))

            intersection = (pred_batch*target_batch).sum(dim=1)*channel_weight
            union = (pred_batch.sum(dim=1) +
                     target_batch.sum(dim=1))*channel_weight

            smooth = 1e-6

            dice = ((2*(intersection+smooth))/(union+smooth)).mean()

            score += dice
        score = score/channels

        likelihood = torch.log1p(torch.cosh(1-score))
        return likelihood
