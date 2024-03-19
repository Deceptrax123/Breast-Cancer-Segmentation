import torch
from torch import nn
import torch.nn.functional as f

# input : BXCXHXW
# target : BXCXHXW


def overall_dice_score(input, target):

    probs = f.sigmoid(input)

    predictions = torch.where(probs > .5, 1, 0)

    pred_bflat = predictions.view(predictions.size(
        0), predictions.size(1)*predictions.size(2)*predictions.size(3))
    target_bflat = target.view(target.size(
        0), target.size(1)*target.size(2)*target.size(3))

    intersection = (pred_bflat*target_bflat).sum(dim=1)
    union = pred_bflat.sum(dim=1)+target_bflat.sum(dim=1)

    smooth = 1e-6

    dice = (2*(intersection+smooth)/(union+smooth)).mean()

    return dice
