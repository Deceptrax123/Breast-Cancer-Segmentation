import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from metrics import overall_dice_score
import matplotlib.pyplot as plt
from Model.Unet.model import CombinedModel
from Model.SegNet.model import SegNetModel
from dotenv import load_dotenv
from cancer_dataset import TestBreastCancerDataset
import torch.nn.functional as f
import os
from PIL import Image
import random
import cv2


def evaluate():
    model.eval()
    prec = 0
    rec = 0
    dice = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            preds, probs = model(x)

            dice += overall_dice_score(preds, y).item()
            prec += precision(preds, y).item()
            rec += recall(preds, y).item()

        print("Recall: ", rec/(i+1))
        print("Precision: ", prec/(i+1))
        print("Dice Score: ", dice/(i+1))


if __name__ == '__main__':
    weights = torch.load(
        "weights/best_segnet_dice.pth", map_location='cpu')

    load_dotenv('.env')

    model = SegNetModel(4)

    # load model with trained weights
    model.load_state_dict(weights)

    path = os.getenv("Images_Test")
    mask_path = os.getenv("Masks_Test")

    xpaths = sorted(os.listdir(path))
    ypaths = sorted(os.listdir(mask_path))

    # Remove fragments and .xmls
    xps = list()
    for i in xpaths:
        if i[1] not in ['_'] and '.xml' not in i:
            xps.append(i)

    yps = list()
    for i in ypaths:
        if i[1] not in ['_'] and '.xml' not in i:
            yps.append(i)

    ds_paths = list(zip(xps, yps))

    test_dataset = TestBreastCancerDataset(paths=ds_paths)
    params = {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 0
    }

    test_loader = DataLoader(test_dataset, **params)

    # Metrics
    precision = BinaryPrecision()
    recall = BinaryRecall()

    evaluate()
