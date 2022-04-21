import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks
from vivit import *
from utils import *
import pickle

import hades

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

print(device)

# Hyperparameters & Important Variables
#........................................................................................

data_dir  = "../../data"

subject_ID = 0

pretrained_path   = ""

dataset_size      = 40   # Number of videos in each mini-batch
batch_number      = 0    # Mini-batch number to use 

train_batch_size  = 4
test_batch_size   = int(dataset_size * 0.2)


use_pretrained   = False   

# Model Hyperparameters
image_size  = 240
patch_size  = 16
num_classes = 2
num_frames  = 134
dim         = 128 # Embedding Size
num_epochs  = 30


### DataLoader Setup
#...........................................................................................................................................................................

from torch.utils.data import random_split, DataLoader, TensorDataset
from torchvision import transforms

class ViViTDataModule(pl.LightningDataModule):
    def __init__(self, include_audio=False):
    # Define required parameters here
        super().__init__()

        with open('../../data/features-1.pkl','rb') as f: 
          data = pickle.load(f)

        self.videos, self.labels = 0, 0

        self.transform = transforms.Compose([
          transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
    
    def prepare_data(self):
    # Define steps that should be done on only one GPU, like getting data.   
      self.videos = torch.FloatTensor(self.videos)
      #self.videos = self.videos.permute(0, 1, 4, 2, 3)
      self.videos = self.transform(self.videos)
      self.labels = torch.Tensor(self.labels).long()
      
      self.full_dataset = TensorDataset(self.videos, self.labels)
      del self.videos, self.labels

      self.length = len(self.full_dataset)

    def setup(self, stage=None):
    # Define steps that should be done on every GPU, like splitting data, applying transform etc.
      self.train_dataset, self.val_dataset = random_split(self.full_dataset, [int(self.length * 0.8), int(self.length * 0.2)], generator=torch.Generator().manual_seed(42))

    
    def train_dataloader(self):
    # Return DataLoader for Training Data here optimizer
        return DataLoader(self.train_dataset, batch_size=train_batch_size)

    def val_dataloader(self):
    # Return DataLoader for Validation Data here
        return DataLoader(self.val_dataset, batch_size=test_batch_size)

    def test_dataloader(self):
        # Return DataLoader for Testing Data here
        pass

#...........................................................................................................................................................................
### TRAINING PHASE

# If tensorboard is to be used in a .ipynb file
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs

from pytorch_lightning.loggers import TensorBoardLogger
torch.autograd.set_detect_anomaly(True)

train_losses.clear()
val_losses.clear()

# Setup your training
model = ViViT(image_size, patch_size, num_classes, num_frames, dim)
model = model.to(device)

# Load Pre-trained Model

if(use_pretrained):
    checkpoint = torch.load(pretrained_path)
    model.load_state_dict(checkpoint['state_dict'])

parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)

# Setup DataLoader
data_module = ViViTDataModule(include_audio=include_audio)

# Logging
logger = TensorBoardLogger(save_dir = os.getcwd(), name = "lightning_logs")

# Training Code
trainer = pl.Trainer(logger=logger, gpus=1, max_epochs=num_epochs)
torch.autograd.set_detect_anomaly(True)
trainer.fit(model, data_module)

# trainer.validate(model, datamodule=data_module)
#...........................................................................................................................................................................