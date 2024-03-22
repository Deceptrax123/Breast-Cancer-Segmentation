import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import mps, cpu
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from metrics import overall_dice_score
from losses import DiceLoss
from cancer_dataset import BreastCancerDataset
from torchmetrics.classification import BinaryRecall, BinaryPrecision
from dotenv import load_dotenv
from Model.Unet.model import CombinedModel
from Model.SegNet.model import SegNetModel
import wandb
from torchsummary import summary
import os


def compute_weights(y_sample):
    channels = 1
    counts = list()
    for i in range(channels):
        spectral_region = y_sample[:, i, :, :]

        ones = (spectral_region == 1.).sum()

        if ones == 0:
            ones = np.inf
        counts.append(ones)

    total_pixels = y_sample.size(0)*512*512

    counts = np.array(counts)
    weights = counts/total_pixels

    inverse = (1/weights)
    inverse = inverse.astype(np.float32)
    return inverse


def train_step():
    epoch_loss = 0
    dice_score = 0
    rec = 0
    prec = 0

    for step, (x_sample, y_sample) in enumerate(train_loader):
        weights = compute_weights(y_sample)
        x_sample = x_sample.to(device=device)
        y_sample = y_sample.to(device=device)
        weights = torch.from_numpy(weights).to(device=device)

        # Predictionns-Forward propagation
        predictions, probs = model(x_sample)

        # Backpropagation
        model.zero_grad()

        # Loss function
        # More stable than BCELoss()
        loss = DiceLoss(weights=weights)

        loss_value = loss(predictions, y_sample)

        loss_value.backward()
        model_optimizer.step()

        # Add losses
        epoch_loss += loss_value.item()

        # Add Metrics
        dice_score += overall_dice_score(predictions, y_sample).item()

        # Detach tensors from GPU
        probs = probs.to(device='cpu')
        y_sample = y_sample.to(device='cpu')

        prec += precision(probs, y_sample).item()
        rec += recall(probs, y_sample).item()

        # Memory
        del x_sample
        del y_sample
        del predictions

        mps.empty_cache()

    return epoch_loss/train_steps, dice_score/train_steps, prec/train_steps, rec/train_steps


def test_step():
    epoch_loss = 0
    dice_score = 0
    prec = 0
    rec = 0

    for step, (x_sample, y_sample) in enumerate(test_loader):
        weights = compute_weights(y_sample)
        x_sample = x_sample.to(device=device)
        y_sample = y_sample.to(device=device)
        weights = torch.from_numpy(weights).to(device=device)

        # Predictionns-Forward propagation
        predictions, probs = model(x_sample)

        loss = DiceLoss(weights=weights)
        loss_value = loss(predictions, y_sample)

        # Add losses
        epoch_loss += loss_value.item()
        dice_score += overall_dice_score(predictions, y_sample).item()

        # Detach tensors from GPU
        probs = probs.to(device='cpu')
        y_sample = y_sample.to(device='cpu')

        prec += precision(probs, y_sample).item()
        rec += recall(probs, y_sample).item()

        # Memory
        del x_sample
        del y_sample
        del predictions

        mps.empty_cache()

    return epoch_loss/test_steps, dice_score/test_steps, prec/test_steps, rec/test_steps


def training_loop():

    for epoch in range(num_epochs):
        model.train(True)  # switch model to train mode

        train_loss, train_dice, train_precision, train_recall = train_step()
        model.eval()

        with torch.no_grad():
            test_loss, test_dice, test_precision, test_recall = test_step()

            print("Epoch: ", epoch+1)
            print("Train Loss: ", train_loss)
            print("Train Dice: ", train_dice)
            print("Train Precision: ", train_precision)
            print("Train Precision: ", train_precision)
            print("Test Precision: ", test_precision)

            print("Test Loss: ", test_loss)
            print("Test Dice: ", test_dice)
            print("Test Precision: ", test_precision)
            print("Test Recall: ", test_recall)

            wandb.log({
                "Train Loss": train_loss,
                "Test Loss": test_loss,
                "Train Precision": train_precision,
                "Train Recall": train_recall,
                "Train Dice": train_dice,
                "Test Precision": test_precision,
                "Test Recall": test_recall,
                "Test Dice": test_dice
            })

            # checkpoints
            if ((epoch+1) % 10 == 0 and epoch >= 1000):
                torch.save(model.state_dict(),
                           'weights/segnet_diceloss/model{epoch}.pth'.format(epoch=epoch+1))


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

    wandb.init(
        project='Loss-Functions-Breast-Cancer',
        config={
            "arcitecture": "DL Models",
            "dataset": "Breast cancer dataset"
        }
    )

    # Store paths in a list
    image_env_paths = os.getenv("Images")
    mask_env_paths = os.getenv("Mask")

    # Get the TIF files
    image_paths = os.listdir(image_env_paths)
    mask_paths = os.listdir(mask_env_paths)

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

    # Map the paths as X and Y
    paths_dataset = list(
        zip(image_paths_new, mask_paths_new))

    # Train-test split
    train, test = train_test_split(paths_dataset, train_size=0.90)

    # Call the dataset
    train_set = BreastCancerDataset(train)
    test_set = BreastCancerDataset(test)

    # Create a dataloader
    train_loader = DataLoader(dataset=train_set, **params)
    test_loader = DataLoader(dataset=test_set, **params)

    # Device
    device = torch.device("mps")

    # Hyperparameters
    lr = 0.001
    num_epochs = 5000

    model = SegNetModel(4).to(device=device)

    # Metrics
    precision = BinaryPrecision()
    recall = BinaryRecall()

    model_optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.001)

    train_steps = (len(train_set)+params['batch_size'])//params['batch_size']
    test_steps = (len(test_set)+params['batch_size'])//params['batch_size']

    training_loop()
