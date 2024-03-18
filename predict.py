import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from Model.model import CombinedModel
from dotenv import load_dotenv
import torch.nn.functional as f
import os
from PIL import Image
import random
import cv2


if __name__ == '__main__':
    weights = torch.load("weights/model100.pth")

    load_dotenv('.env')

    model = CombinedModel()
    model.eval()

    # load model with trained weights
    model.load_state_dict(weights)

    path = os.getenv("Images")

    xpaths = sorted(os.listdir(path))

    # Remove fragments and .xmls
    xps = list()
    for i in xpaths:
        if i[1] not in ['_'] and '.xml' not in i:
            xps.append(i)
    # select a random image
    random_img_path = random.choice(xps)

    img = Image.open(path+random_img_path)

    tens_transform = T.Compose([
        T.Resize((512, 512)), T.ToTensor(), T.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    img_tensor = tens_transform(img)
    img_tensor = img_tensor.view(1, 3, 512, 512)

    # Get outputs
    predictions = model(img_tensor)
    predictions_probabs = f.sigmoid(predictions)

    # Get the Mask
    preds = torch.argmax(predictions_probabs, dim=1)
    predictions_probs = torch.zeros_like(
        predictions_probabs).scatter_(1, preds.unsqueeze(1), 1.)

    predictions_probs = predictions_probs.permute(0, 2, 3, 1)

    print(predictions_probs.unique())
