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
    weights = torch.load("weights/model1000.pth", map_location='cpu')

    load_dotenv('.env')

    model = CombinedModel()
    model.eval()

    # load model with trained weights
    model.load_state_dict(weights)

    path = os.getenv("Images")
    mask_path = os.getenv("Mask")

    xpaths = sorted(os.listdir(path))

    # Remove fragments and .xmls
    xps = list()
    for i in xpaths:
        if i[1] not in ['_'] and '.xml' not in i:
            xps.append(i)
    # select a random image
    random_img_path = random.choice(xps)

    img = Image.open(path+random_img_path)
    mask = Image.open(mask_path+random_img_path)

    mask = mask.resize((512, 512), Image.LANCZOS)

    tens_transform = T.Compose([
        T.Resize((512, 512)), T.ToTensor(), T.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    img_tensor = tens_transform(img)
    img_tensor = img_tensor.view(1, 3, 512, 512)

    # Get outputs
    predictions = model(img_tensor)
    predictions_probabs = f.sigmoid(predictions)

    # Get the Mask
    preds = torch.where(predictions_probabs > 0.5, 1, 0)

    preds_img = preds.permute(0, 2, 3, 1)
    preds_img_np = preds_img.detach().numpy()

    fig = plt.figure(figsize=(10, 10), dpi=72)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(np.array(img))
    ax1.set_title("Input")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(np.array(mask))
    ax2.set_title("Ground truth")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(preds_img_np[0])
    ax3.set_title("Predicted")

    plt.show()
