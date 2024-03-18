import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import mps, cpu
import gc
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy(
        'file_system')  # Ensures that there is no file limit
    # while processing dataset
    load_dotenv('.env')  # Path to env file
    # Get data, batch data, run models

    # Get Data-Dataset
    # Batch-Dataloader
    # Run-Models

    params = {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 0
    }

    # Store paths in a list
    image_paths = os.getenv("Images")
    mask_paths = os.getenv("Mask")

    # Get the TIF files
    image_paths = os.listdir(image_paths)
    mask_paths = os.listdir(mask_paths)

    # Filter out fragmented files
    img_paths_new = list()
    mask_paths_new = list()

    for i in image_paths:
        if i[1] not in ['_'] and '.xml' not in i:
            img_paths_new.append(i)
    for j in mask_paths:
        if j[1] not in ['_'] and '.xml' not in j:
            mask_paths_new.append(j)

    image_paths_new = sorted(img_paths_new)
    mask_paths_new = sorted(mask_paths_new)
