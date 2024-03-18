import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import os
from dotenv import load_dotenv
from PIL import Image


class BreastCancerDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)  # Returns length of dataset

    def __getitem__(self, index):  # Gets each image from path
        sample = self.paths[index]
        load_dotenv('.env')
        x_env_path = os.getenv("Images")
        y_env_path = os.getenv("Mask")

        x_path, y_path = sample[0], sample[1]

        # Read image from path
        x_img, y_img = Image.open(
            x_env_path+x_path), Image.open(y_env_path+y_path)

        # Preprocessing->Convert to Tensor
        process_function_x = T.Compose([
            T.Resize(size=(512, 512)), T.ToTensor(), T.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        process_function_y = T.Compose([
            T.Resize(size=(512, 512)), T.ToTensor()
        ])

        x_tensor = process_function_x(x_img)
        y_tensor = process_function_y(y_img)

        return x_tensor, y_tensor
