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
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp

if __name__ == '__main__':
    weights = torch.load("weights/model1000.pth", map_location='cpu')

    load_dotenv('.env')

    model = CombinedModel()
    model.eval()
    
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
    
    fig = plt.figure(figsize=(10, 10), dpi=72)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.imshow(np.array(img))
    ax1.set_title("Input")